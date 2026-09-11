import cv2
import numpy as np
import matplotlib.pyplot as plt
import openai
from openai import OpenAI
import base64
import json
import pyrealsense2 as rs
import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
import logging
import re
import os
import threading
import time

# 配置日志记录器
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 主要功能说明：
# 步骤1: 通过命令行获取用户输入的prompt，例如：
#   - "桌子上有哪些东西？"
#   - "把香蕉放进盒子里"
# 步骤2: 根据prompt类型路由到对应函数：
#   A. 如果是查询类问题（如"桌子上有哪些东西？"），则只进行显示和打印
#   B. 如果是抓取类指令（如"把香蕉放进盒子里"），则调用graspnet进行抓取

class VLM_GRASP:
    def __init__(self):
        # API密钥配置，从环境变量获取
        self.API_KEY = os.getenv("DASHSCOPE_API_KEY")
        if not self.API_KEY:
            raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")
    
        # LLM类型选择：本地使用ollama，API使用阿里云
        self.LLM_TYPE = ["LOCAL","API"][1] 
        # 本地ollama模型配置
        self.LOCAL_OLLAMA_MODEL = "qwen2.5:7b"
        # API LLM模型配置
        # 文本路由模型。deepseek-v4 不是有效的 DashScope 模型 ID。
        # 可通过环境变量覆盖，便于以后在不改代码的情况下切换模型。
        self.API_LLM_MODEL = os.getenv("DASHSCOPE_LLM_MODEL", "qwen-plus")
        # VLM模型配置
        # 参考文档：https://bailian.console.aliyun.com/?tab=doc#/doc/?type=model&url=https%3A%2F%2Fhelp.aliyun.com%2Fdocument_detail%2F2845871.html&renderType=iframe
        # 由于多模态模型对硬件要求较高，直接使用API
        self.API_VLM_MODEL = os.getenv(
            "DASHSCOPE_VLM_MODEL",
            "qwen3-vl-plus",
        )
        # 检测模型配置，可根据需要在此处修改
        self.DETECTION_MODEL = [self.API_VLM_MODEL,'grounding-dino-base'][0]


        # 输出相关使用的模型信息
        if self.LLM_TYPE == "LOCAL":
            logger.info(f"LLM使用的是本地ollama模型：{self.LOCAL_OLLAMA_MODEL}")
        else:
            logger.info(f"LLM使用的是阿里云API模型：{self.API_LLM_MODEL}")
        logger.info(f"VLM使用的是阿里云API模型：{self.API_VLM_MODEL}")
        logger.info(f"检测模型使用的是：{self.DETECTION_MODEL}")
        
        # RealSense配置
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.frame_w, self.frame_h = 1280, 720
        self.config.enable_stream(rs.stream.depth, self.frame_w, self.frame_h, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, self.frame_w, self.frame_h, rs.format.bgr8, 30)
        self.align = rs.align(rs.stream.color)
        
        # 视频流显示相关
        self.running = True
        self.scale = 0.7
        self.current_color = None
        self.current_depth_frame = None # Store the depth frame object
        self.display_lock = threading.Lock()
        
        # 检测框信息存储
        self.detection_boxes = []  # 格式：[(x1,y1,x2,y2,label,color)]
        self.detection_lock = threading.Lock()
        
        # 当前模式
        self.current_mode = "Ready"
        self.mode_lock = threading.Lock()
        
        self.display_thread = threading.Thread(target=self.realsense_video)
        self.display_thread.daemon = True
        self.display_thread.start()

    def set_mode(self, mode):
        """设置当前模式"""
        with self.mode_lock:
            self.current_mode = mode

    def get_mode(self):
        """获取当前模式"""
        with self.mode_lock:
            return self.current_mode

    def clear_detection_boxes(self):
        """清除当前的检测框"""
        with self.detection_lock:
            self.detection_boxes = []

    def add_detection_box(self, x1, y1, x2, y2, label, color=(0,255,0)):
        """添加一个检测框到显示列表
        参数:
            x1, y1: 左上角坐标
            x2, y2: 右下角坐标
            label: 标签文本
            color: 框和文本的颜色，默认绿色
        """
        with self.detection_lock:
            self.detection_boxes.append((x1, y1, x2, y2, label, color))

    def vlm_bbox_to_pixels(self, xyxy, image_width, image_height):
        """把Qwen3-VL的0~1000相对坐标转换成原图像素坐标。"""
        try:
            if (
                not isinstance(xyxy, (list, tuple))
                or len(xyxy) != 2
                or len(xyxy[0]) != 2
                or len(xyxy[1]) != 2
            ):
                raise ValueError("边界框必须是[[x1,y1],[x2,y2]]")

            x1_norm, y1_norm = map(float, xyxy[0])
            x2_norm, y2_norm = map(float, xyxy[1])

            # Qwen3-VL使用保持宽高比的原图特征，但坐标输出位于1000×1000参考网格。
            x1 = round(x1_norm * image_width / 1000.0)
            y1 = round(y1_norm * image_height / 1000.0)
            x2 = round(x2_norm * image_width / 1000.0)
            y2 = round(y2_norm * image_height / 1000.0)

            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))
            x1 = max(0, min(image_width - 1, x1))
            y1 = max(0, min(image_height - 1, y1))
            x2 = max(0, min(image_width - 1, x2))
            y2 = max(0, min(image_height - 1, y2))

            if x2 <= x1 or y2 <= y1:
                raise ValueError("换算后的边界框面积为0")

            pixel_box = [x1, y1, x2, y2]
            logger.info(
                "VLM边界框换算：relative_1000=%s -> pixels=%s，image=%dx%d",
                xyxy,
                pixel_box,
                image_width,
                image_height,
            )
            return pixel_box
        except (TypeError, ValueError) as exc:
            logger.error("无法解析VLM边界框 %r：%s", xyxy, exc)
            return None

    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        pass

    def realsense_video(self):
        """实时显示视频流的线程函数"""
        pipe = self.pipeline
        cfg = self.config
        align = self.align

        try:
            pipe.start(cfg)
            cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Detection', int(720*self.scale*2), int(720*self.scale))
            cv2.setMouseCallback('Detection', self.mouse_callback)

            while self.running:
                t_start = time.time()
                frames = pipe.wait_for_frames()
                aligned_frames = align.process(frames)
                
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                if not depth_frame or not color_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                
                with self.display_lock:
                    self.current_color = color_image.copy()
                    self.current_depth_frame = depth_frame
                
                # 为显示准备一个新的副本
                display_color_image = color_image.copy()
                
                # 在显示图像上绘制所有检测框
                with self.detection_lock:
                    for box in self.detection_boxes:
                        x1, y1, x2, y2, label, color = box
                        cv2.rectangle(display_color_image, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(display_color_image, label, (x1, y1-5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                # 深度图转换为彩色图
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(np.asanyarray(depth_frame.get_data()), alpha=0.14), 
                    cv2.COLORMAP_JET
                )
                
                # 裁剪图像显示中间区域
                h_color, w_color = display_color_image.shape[:2]
                h_depth, w_depth = depth_colormap.shape[:2]
                
                # 计算裁剪起点以获取中间720x720区域
                start_x_color = max(0, (w_color - 720) // 2)
                start_y_color = max(0, (h_color - 720) // 2)
                start_x_depth = max(0, (w_depth - 720) // 2)
                start_y_depth = max(0, (h_depth - 720) // 2)
                
                # 裁剪获取中间区域
                display_color_image_cropped = display_color_image[start_y_color:start_y_color + 720, 
                                                                start_x_color:start_x_color + 720]
                depth_colormap_cropped = depth_colormap[start_y_depth:start_y_depth + 720, 
                                                       start_x_depth:start_x_depth + 720]
                
                # FPS计算和显示
                fps = 1.0 / (time.time() - t_start)
                cv2.putText(display_color_image_cropped, f"FPS: {fps:.2f}", (20, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                
                # 显示当前模式
                current_mode = self.get_mode()
                cv2.putText(display_color_image_cropped, f"Mode: {current_mode}", (20, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

                # 水平拼接裁剪后的颜色图和深度图
                images = np.hstack((display_color_image_cropped, depth_colormap_cropped))
                resized_images = cv2.resize(images, (0,0), fx=self.scale, fy=self.scale)
                cv2.imshow('Detection', resized_images)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                    break
        except Exception as e:
            logger.error(f"RealSense video thread error: {e}")
        finally:
            logger.info("Stopping RealSense pipeline and closing OpenCV windows.")
            pipe.stop()
            cv2.destroyAllWindows()

    def capture_image(self):
        '''
        获取RealSense相机的图像
        返回：
            color_image: 彩色图像
            depth_image: 深度图像
        '''
        with self.display_lock:
            if self.current_color is not None and self.current_depth_frame is not None:
                return self.current_color.copy(), np.asanyarray(self.current_depth_frame.get_data())
        logger.warning("capture_image called when current frames are not available.")
        return None, None

    def get_camera_paras(self):
        '''
        获取RealSense相机的内参矩阵和畸变系数
        返回：
            intr_matrix: 相机内参矩阵
            intr_coeffs: 相机畸变系数
        '''
        with self.display_lock: # Ensure thread-safe access if called from main thread
            if self.current_color is not None: # Check if frames have been processed
                # Attempt to get intrinsics from a live frame if possible (requires pipeline to be active)
                # This part is tricky if the video thread isn't running or hasn't captured frames yet.
                # A robust way would be to capture intrinsics once after pipeline start and store them.
                # For now, let's try to get it from the config or a recent frame
                profiles = self.pipeline.get_active_profile()
                if profiles:
                    stream_profile = profiles.get_stream(rs.stream.color).as_video_stream_profile()
                    intr = stream_profile.get_intrinsics()
                    intr_matrix = np.array([
                        [intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]
                    ]) 
                    intr_coeffs = np.array(intr.coeffs)
                    return intr_matrix, intr_coeffs
        logger.warning("Could not get camera parameters. Pipeline might not be active or no frames captured.")
        # Fallback or raise error
        return np.eye(3), np.zeros(5) 

    def saveImg(self, color_image, depth_image):
        '''
        保存图像到指定路径
        参数：
            color_image: 彩色图像
            depth_image: 深度图像
        返回：
            colorFileName: 彩色图像保存路径
            depthFileName: 深度图像保存路径
        '''
        if color_image is None or depth_image is None:
            logger.error("Cannot save None images.")
            return None, None
        os.makedirs('./VLM_related/realsense_captured/', exist_ok=True)
        np.save('./VLM_related/realsense_captured/color.npy', color_image)
        np.save('./VLM_related/realsense_captured/depth.npy', depth_image)
        colorFileName = './VLM_related/realsense_captured/color.jpg'
        depthFileName = './VLM_related/realsense_captured/depth.png'
        cv2.imwrite(colorFileName, color_image)
        # For depth, ensure it's a savable format (e.g., 16-bit grayscale)
        if depth_image.dtype == np.uint16:
            cv2.imwrite(depthFileName, depth_image) 
        else:
            # Convert if necessary, e.g. if it was scaled for display
            cv2.imwrite(depthFileName, cv2.convertScaleAbs(depth_image, alpha=(255.0/np.max(depth_image) if np.max(depth_image)>0 else 1.0)))
        return colorFileName, depthFileName

    def ask_LLM(self, content):
        '''
        调用大语言模型，支持本地Ollama和API两种模式
        参数：
            content: 输入内容
        返回：
            result: 模型返回的结果
        '''
        if self.LLM_TYPE == "API":
            client = OpenAI(
                api_key=self.API_KEY,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            try:
                # 向大模型发起请求
                completion = client.chat.completions.create(
                    model=self.API_LLM_MODEL,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": content,
                                }
                            ],
                        },
                    ],
                )
            except openai.APIStatusError as exc:
                logger.error(
                    "LLM API请求失败：HTTP %s，模型=%s，响应=%s",
                    exc.status_code,
                    self.API_LLM_MODEL,
                    exc.response.text[:500],
                )
                return None
            except openai.APIConnectionError as exc:
                logger.error("LLM API连接失败：%s", exc)
                return None
            except Exception:
                logger.exception("LLM调用发生未预期异常")
                return None
            # 解析大模型返回结果
            result = completion.choices[0].message.content.strip()
        else:  # 本地模式
            from ollama import chat
        
            # 使用ollama进行普通对话
            response = chat(
                model=self.LOCAL_OLLAMA_MODEL,
                messages=[{'role': 'user', 'content': content}]
            )
            result = response.message.content.strip()
            
        if not result:
            logger.error("LLM返回了空内容")
            return None

        # 如果有<think></think>标签，则去掉<think>和</think>之间的内容
        result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL)
        # 清理可能留下的多余空白行
        result = re.sub(r'\n\s*\n', '\n', result).strip()
        # print(f"返回结果：{result}")
        return result

    def route2func(self, prompt):
        '''
        根据用户输入的prompt路由到对应的功能函数
        参数：
            prompt: 用户输入的指令
        返回：
            func_name: 函数名列表
        '''
        # 常见的场景查询不需要额外调用一次LLM，避免增加延迟和故障点。
        normalized_prompt = re.sub(r"[\s，。！？?！]", "", prompt)
        list_patterns = ("桌子上有什么", "桌上有什么", "有哪些东西", "有什么东西")
        action_patterns = ("抓", "拿", "放", "移动", "吸取", "夹取")
        if (
            any(pattern in normalized_prompt for pattern in list_patterns)
            and not any(pattern in normalized_prompt for pattern in action_patterns)
        ):
            return ["list_objs"]

        SYSTEM_PROMPT = '''
        我将给你一个字符串。你帮我输出函数名：
        1. 如果字符串是：桌子上有哪些东西？有什么？桌子上有什么等类似咨询问题？你就输出函数名：["list_objs"]
        2. 如果字符串是：把最大/黄色的/最小的/白色的物体放进盒子等类似抓取动作。你就输出函数名和描述物体的词：["grasp_obj","描述词"]，如：["grasp_obj","粉红色的遥控器"]、["grasp_obj","最大的苹果"]、["grasp_obj","香蕉右侧的胶棒"]
        3. 如果是其他问题，你就输出函数名：["other"]
        注意：
        1. 如果是list_objs，只需要输出函数名本身，不需要任何其他描述文字
        2. 如果是grasp_obj，需要输出函数名和描述物体的词，如：["grasp_obj","描述词"]，不要输出任何其他描述文字
        3. 不要漏掉[]
        4. 外部不要加```这种引号
        5. 最终格式是：["函数名"]或["函数名","描述词"]，没有外围文字

        我现在给你的字符串是：'''

        # 调用LLM进行解析
        func_name = self.ask_LLM(SYSTEM_PROMPT + prompt)
        if not func_name:
            return None

        # 模型被要求返回标准JSON数组。不要使用eval执行模型生成的文本。
        try:
            parsed = json.loads(func_name)
        except json.JSONDecodeError:
            logger.error("LLM函数路由结果不是有效JSON：%s", func_name[:500])
            return None

        if not isinstance(parsed, list):
            logger.error("LLM函数路由结果不是列表：%r", parsed)
            return None
        return parsed

    def ask_vlm(self, img_path, PROMPT):
        '''
        调用视觉语言模型进行图像理解和目标检测
        参数：
            img_path: 图像路径
            PROMPT: 提示词
        返回：
            result: 模型返回的结果
        '''
        # 系统提示词
        SYSTEM_PROMPT = '''
        我将给你一张图片，以及一个指令：
        1. 如果指令是："list_objs"，你就：
        列出图中所有你能看到的物体，整理成JSON格式，格式如下：

        {
        "function":"list_objs",
        "objs":[
        ["类别名称",[左上角相对坐标x,左上角相对坐标y],[右下角相对坐标x,右下角相对坐标y]],
        ["类别名称",[左上角相对坐标x,左上角相对坐标y],[右下角相对坐标x,右下角相对坐标y]],
        ]
        }

        如：
        {
        "function":"list_objs",
        "objs":[
        ["apple",[100,100],[300,300]],
        ["banana",[120,100],[320,300]],
        ["apple",[120,120],[320,320]],
        ["banana",[420,400],[820,820]],
        ]
        }


        2. 如果指令是："grasp_obj"，你就找出最符合的一个目标输出，注意只有一个目标：
        {{
          "function":"grasp_obj",
          "className":"类别名称",
          "xyxy":[[左上角相对坐标x,左上角相对坐标y],[右下角相对坐标x,右下角相对坐标y]]
        }}
        如：指令是 把最大的橘子放进盒子里，你找到最大的橘子的坐标：
        {{
          "function":"grasp_obj",
          "className":"Orange",
          "xyxy":[[102,505],[324,860]],
        }}

        3. 如果指令是："descibe_obj"。你就描述能看到的所有物体（不要忽略了一些小的物体），用逗号,分开，如
        {
        "function":"descibe_obj",
        "objs":["apple","banana","orange","milk box","mouse"]
        }


        注意：
        1. 只需要输出JSON本身，不需要任何其他数据，尤其是JSON前后的```符号
        2. 不要少了function
        3. 所有xyxy坐标必须使用0到1000的相对坐标系：原图左上角是(0,0)，
           原图右下角是(1000,1000)。不要输出原图像素坐标。

        现在指令是： '''

        # 初始化OpenAI客户端
        client = OpenAI(
            api_key= self.API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        
        # 将图像编码为base64格式
        with open(img_path, 'rb') as image_file:
            image = 'data:image/jpeg;base64,' + base64.b64encode(image_file.read()).decode('utf-8')
        try:
            # 向大模型发起请求
            completion = client.chat.completions.create(
                model=self.API_VLM_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": SYSTEM_PROMPT + PROMPT,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image,
                                },
                            },
                        ],
                    },
                ],
            )
        except openai.APIStatusError as exc:
            logger.error(
                "VLM API请求失败：HTTP %s，模型=%s，响应=%s",
                exc.status_code,
                self.API_VLM_MODEL,
                exc.response.text[:500],
            )
            return None
        except openai.APIConnectionError as exc:
            logger.error("VLM API连接失败：%s", exc)
            return None
        except Exception:
            logger.exception("VLM调用发生未预期异常")
            return None
        
        # 解析大模型返回结果
        result = completion.choices[0].message.content.strip()
        
        return result

    def groundingDINO(self,img_path, classes):
        '''
        使用Grounding DINO模型进行目标检测
        参数：
            img_path: 图像路径
            classes: 目标类别列表
        返回：
            results: 检测结果
        '''
        model_id = "./VLM_related/grounding-dino-base"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

        image = Image.open(img_path)
        # 构建文本查询，注意：文本查询需要小写并以点号结尾
        text = ""
        for i in classes:
          text += i + "."

        inputs = processor(images=image, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.3,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]]
        )
        return results

    def jsonOriganiner(self,json_str,PROMPT):
        '''
        将模型返回的字符串整理成标准JSON格式
        参数：
            json_str: 原始JSON字符串
            PROMPT: 提示词
        返回：
            json_data: 整理后的JSON数据
        '''
        if not json_str:
            logger.error("没有可整理的VLM结果")
            return {}

        # VLM通常已经按照提示返回JSON。优先直接解析，避免重复调用文本模型。
        cleaned_json = re.sub(r"^\s*```(?:json)?\s*", "", json_str, flags=re.IGNORECASE)
        cleaned_json = re.sub(r"\s*```\s*$", "", cleaned_json)
        json_start = cleaned_json.find("{")
        json_end = cleaned_json.rfind("}")
        if json_start != -1 and json_end > json_start:
            try:
                parsed_json = json.loads(cleaned_json[json_start:json_end + 1])
                if isinstance(parsed_json, dict):
                    return parsed_json
            except json.JSONDecodeError:
                logger.info("VLM原始结果不是标准JSON，尝试使用文本模型整理")

        SYSTEM_PROMPT = '''
        我将给你一个字符串，和一个指令。
        如果指令是："list_objs"，你就将字符串整理成如下JSON格式：

        {
          "function":"list_objs",
          "objs":[
            ["className",[左上角相对坐标x,左上角相对坐标y],[右下角相对坐标x,右下角相对坐标y]],
            ["className",[左上角相对坐标x,左上角相对坐标y],[右下角相对坐标x,右下角相对坐标y]],
          ]
        }

        如：
        {
          "function":"list_objs",
          "objs":[
            ["apple",[100,100],[300,300]],
            ["banana",[120,100],[320,300]],
            ["apple",[120,120],[320,320]],
            ["banana",[420,400],[820,820]],
          ]
        }


        2. 如果指令是："grasp_obj"，你就输出：
        {{
          "function":"grasp_obj",
          "className":"类别名称",
          "xyxy":[[左上角相对坐标x,左上角相对坐标y],[右下角相对坐标x,右下角相对坐标y]]
        }}
        如：
        {{
          "function":"grasp_obj",
          "className":"Orange",
          "xyxy":[[102,505],[324,860]],
        }}

        3. 如果指令是："descibe_obj"。你就输出：
        {
        "function":"descibe_obj",
        "objs":["类别名称","类别名称","类别名称","类别名称","类别名称"]
        }
        如：
        {
        "function":"descibe_obj",
        "objs":["apple","banana","orange","milk box","mouse"]
        }


        注意：
        1. 只需要输出JSON本身，不需要任何其他数据，尤其是JSON前后的```符号
        2. 类别要求是英文
        3. xyxy使用0到1000的相对坐标系，原图左上角为(0,0)，右下角为(1000,1000)

        我现在给你的字符串是：'''
        # 调用LLM进行JSON格式整理
        json_data = self.ask_LLM(SYSTEM_PROMPT + json_str + "。\n指令是：" + PROMPT)

        if not json_data:
            return {}
        try:
            return json.loads(json_data)
        except json.JSONDecodeError:
            logger.error("JSON整理结果无法解析：%s", json_data[:500])
            return {}

    
    def saveDataAndCallGraspnet(self,data_list):
        '''
        保存数据并调用GraspNet进行抓取
        参数：
            data_list: 数据列表
        '''
        pass
      
    def __del__(self):
        """清理资源"""
        self.running = False

# 主程序入口
if __name__ == '__main__':
    demo = VLM_GRASP()
    try:
        # 等待视频流初始化
        time.sleep(2)

        while demo.running:
            # 设置为等待输入模式
            demo.set_mode("Ready")
            
            # 获取用户输入
            prompt = input("请输入prompt，可以输入：桌子上有哪些东西？有什么？桌子上有什么等类似咨询问题，"
                         "也可以输入：把香蕉放进盒子里等类似抓取动作(q to quit): ")
            
            # 检查是否退出
            if prompt.lower() == 'q':
                demo.running = False
                break

            # 解析用户指令
            logger.info("LLM开始解析函数名...")
            func_name_data = demo.route2func(prompt)

            # 验证解析结果
            if not func_name_data or not isinstance(func_name_data, list) or len(func_name_data) == 0:
                logger.warning(f"无效的函数名解析结果: {func_name_data}")
                continue
            
            func_name = func_name_data[0]

            # 获取相机图像
            logger.info("相机开始获取图像...")
            color_image, depth_image = demo.capture_image()

            # 验证图像获取
            if color_image is None or depth_image is None:
                logger.error("无法从相机获取图像，跳过当前指令。")
                continue

            # 保存图像
            colorFilePath, depthFilePath = demo.saveImg(color_image, depth_image)
            if colorFilePath is None:
                logger.error("保存图像失败，跳过当前指令。")
                continue

            # 清除上一次的检测框
            demo.clear_detection_boxes()
            
            # 准备用于保存的图像副本
            img_for_drawing = color_image.copy()
            image_height, image_width = img_for_drawing.shape[:2]
            
            # 初始化检测结果变量
            vlm_list_objs_data = None
            ground_dino_results = None
            grasp_obj_data = None
            grasp_obj_box_coords = None

            # 根据功能类型执行相应操作
            if func_name == "list_objs":
                # 设置为列表模式
                demo.set_mode("List")
                
                logger.info("LLM解析成功，开始执行list_objs函数，即列出所有物体")
                
                # 使用VLM模型进行检测
                if demo.DETECTION_MODEL == demo.API_VLM_MODEL:
                    logger.info("即将使用VLM列出所有物体")
                    json_str_vlm = demo.ask_vlm(colorFilePath, "list_objs")
                    vlm_list_objs_data = demo.jsonOriganiner(json_str_vlm, "list_objs")
                    
                    # 处理VLM检测结果
                    if vlm_list_objs_data.get('function') == 'list_objs':
                        for obj in vlm_list_objs_data.get('objs', []):
                            obj_type, lxy, rxy = obj[:3]
                            box_coords = demo.vlm_bbox_to_pixels(
                                [lxy, rxy],
                                image_width,
                                image_height,
                            )
                            if box_coords is None:
                                continue
                            x1, y1, x2, y2 = box_coords
                            
                            # 添加检测框
                            demo.add_detection_box(x1, y1, x2, y2, obj_type, (0,255,0))
                            
                            # 在结果图像上绘制
                            cv2.rectangle(img_for_drawing, (x1, y1), (x2, y2), (0,255,0), 2)
                            cv2.putText(img_for_drawing, f"{obj_type}", (x1, y1), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                
                # 使用Grounding DINO模型进行检测
                else:
                    logger.info("先使用VLM获取场景中的物体类别列表")
                    json_str_desc = demo.ask_vlm(colorFilePath, "descibe_obj")
                    json_data_desc = demo.jsonOriganiner(json_str_desc, "descibe_obj")
                    class_names = json_data_desc.get('objs', [])
                    
                    if class_names:
                        logger.info(f"调用grounding-dino-base，设置类别为：{class_names}")
                        ground_dino_results = demo.groundingDINO(colorFilePath, class_names)
                        
                        # 处理DINO检测结果
                        for result_set in ground_dino_results:
                            for box, label, score in zip(result_set['boxes'], 
                                                       result_set['labels'], 
                                                       result_set['scores']):
                                box_coords = [int(i) for i in box]
                                x1, y1, x2, y2 = box_coords
                                label_with_score = f"{label} {score:.2f}"
                                
                                # 添加检测框
                                demo.add_detection_box(x1, y1, x2, y2, label_with_score, (0,0,255))
                                
                                # 在结果图像上绘制
                                cv2.rectangle(img_for_drawing, (x1, y1), (x2, y2), (0,255,0), 2)
                                cv2.putText(img_for_drawing, label_with_score, (x1, y1), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            # 执行抓取操作
            elif func_name == "grasp_obj":
                # 设置为抓取模式
                demo.set_mode("Grasp")
                
                # 获取目标物体描述
                desc_content = func_name_data[1] if len(func_name_data) > 1 else ""
                logger.info(f"LLM解析成功，开始执行grasp_obj函数，抓取: {desc_content}")
                
                # 使用VLM进行目标检测
                json_str_grasp = demo.ask_vlm(colorFilePath, f"grasp_obj,{desc_content}")
                grasp_obj_data = demo.jsonOriganiner(json_str_grasp, "grasp_obj")
                
                # 处理检测结果
                if grasp_obj_data and grasp_obj_data.get('function') == 'grasp_obj':
                    xyxy = grasp_obj_data.get('xyxy')
                    if xyxy and len(xyxy) == 2 and len(xyxy[0]) == 2 and len(xyxy[1]) == 2:
                        # 把Qwen3-VL相对坐标换算成原图像素坐标，再增加少量边界。
                        pixel_box = demo.vlm_bbox_to_pixels(
                            xyxy,
                            image_width,
                            image_height,
                        )
                        if pixel_box is None:
                            continue
                        x1 = max(0, pixel_box[0] - 10)
                        y1 = max(0, pixel_box[1] - 10)
                        x2 = min(image_width - 1, pixel_box[2] + 10)
                        y2 = min(image_height - 1, pixel_box[3] + 10)
                        grasp_obj_box_coords = [x1, y1, x2, y2]
                        
                        obj_class = grasp_obj_data.get('className', 'TARGET')
                        
                        # 添加检测框
                        demo.add_detection_box(x1, y1, x2, y2, obj_class, (0,0,255))
                        
                        # 在结果图像上绘制
                        cv2.rectangle(img_for_drawing, (x1, y1), (x2, y2), (0,0,255), 2)
                        cv2.putText(img_for_drawing, obj_class, (x1, y1), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                # 保存抓取信息
                if grasp_obj_box_coords is not None:
                    intr_matrix, _ = demo.get_camera_paras()
                    np.save("./VLM_related/exchange/intr_matrix.npy", intr_matrix)
                    np.save("./VLM_related/exchange/result_box.npy", np.array(grasp_obj_box_coords))
                    logger.info(f"描述为 {desc_content}，边界框为 {grasp_obj_box_coords}。数据已保存。")
                else:
                    logger.warning(f"未能检测到物体: {desc_content} for grasping.")
            
            # 处理无效指令
            elif func_name == "other":
                logger.warning("只能输入有效指令")

            # 裁剪并保存结果图像
            h_draw, w_draw = img_for_drawing.shape[:2]
            start_x_crop = max(0, (w_draw - 720) // 2)
            start_y_crop = max(0, (h_draw - 720) // 2)
            img_for_drawing_cropped = img_for_drawing[start_y_crop:start_y_crop + 720, 
                                                    start_x_crop:start_x_crop + 720]
            cv2.imwrite("result.jpg", img_for_drawing_cropped)
            
            # 记录保存结果
            if func_name == "list_objs" or func_name == "grasp_obj":
                logger.info("result.jpg 已保存 (包含检测框，如有)。")

    except KeyboardInterrupt:
        logger.info("程序被用户中断。")
        demo.running = False
    finally:
        # 清理资源
        logger.info("正在清理资源...")
        demo.running = False 
        if hasattr(demo, 'display_thread') and demo.display_thread.is_alive():
            demo.display_thread.join()
        logger.info("程序已退出。")
