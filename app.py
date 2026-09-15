from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import io
from PIL import Image
import base64

# 1. 初始化 FastAPI 应用
app = FastAPI(title="水面垃圾检测 API")

# 2. 配置跨域允许（让我们的前端网页能访问这个接口）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. 加载我们刚训练好的模型！
# 确保 best.pt 就在这个文件夹下
model = YOLO("best.pt")

# 4. 定义一个 POST 接口，接收图片
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # 读取上传的图片文件
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    
    # 将图片喂给模型推理
    results = model(image)
    
    # 将画了框的结果图转换为 numpy 数组
    res_plotted = results[0].plot() 
    
    # 将结果图编码为 Base64 字符串（方便直接传给前端展示）
    res_img = Image.fromarray(res_plotted[..., ::-1]) # OpenCV 是 BGR 格式，需要转成 RGB
    buffered = io.BytesIO()
    res_img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    # 返回 JSON 数据，包含一张 Base64 的图片
    return {"result_image": f"data:image/jpeg;base64,{img_str}"}