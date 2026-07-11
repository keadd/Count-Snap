# CountSnap

本地运行的拍照数物体工具。上传或拍摄照片后，后端用 OpenCV 找出候选区域，前端可以复核、移动框、缩放框、修改单个区域数量，并导出标注图。

当前重点是“计数”，不是识别物体类别。项目已移除 YOLO/AI 常见物体模式，默认使用颜色分割方案，更适合红色、粉色这类颜色明显的小零件。

## 技术栈

- 前端：React + Vite + TypeScript
- 后端：Python + FastAPI + OpenCV
- 存储：暂无
- 模型：暂无，不依赖训练模型

## 启动后端

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 启动前端

```powershell
cd frontend
npm install
npm run dev -- --host 0.0.0.0
```

打开 Vite 显示的前端地址即可使用。如果要在同一局域网的手机上访问，把 `localhost` 换成电脑的局域网 IP。
