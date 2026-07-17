# CountSnap

> 项目仍在持续开发中，当前功能尚未完善，检测效果和交互流程都会继续调整。

CountSnap 是一个本地运行的拍照计数工具。它可以上传或现场拍摄照片，通过 OpenCV 自动找出候选物体区域，并在前端进行复核、拖动、缩放、修改单个区域数量，最后导出带标注的图片。

项目当前重点是“计数”，不是识别物体类别。默认使用颜色分割和轮廓分析，不依赖 YOLO 或训练模型，更适合红色、粉色、蓝色等颜色明显、彼此相对分开的零件或小物体。

## 功能特性

- 上传图片或使用手机摄像头拍摄图片
- 自动描边并检测大量重复出现、轮廓和大小相近的色块
- 默认要求至少出现 8 个相似区域，也可以手动调整
- 支持基础轮廓检测模式
- 自动显示总数和识别区域数量
- 可手动添加、删除、移动和缩放识别框
- 可单独修改某个区域内的物体数量
- 支持导出带编号和数量标注的图片
- 前端会在检测前压缩大图，减少后端处理压力
- 支持同一局域网内手机访问

## 技术栈

### 前端

- React
- Vite
- TypeScript

### 后端

- Python
- FastAPI
- OpenCV
- NumPy

## 项目结构

```text
Count-Snap/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI 入口和接口定义
│   │   └── detection.py     # OpenCV 检测逻辑
│   ├── requirements.txt
│   └── test-detect.png
├── frontend/
│   ├── src/
│   │   ├── App.tsx          # 主界面和交互逻辑
│   │   ├── main.tsx
│   │   └── styles.css
│   ├── package.json
│   └── vite.config.ts
├── .gitignore
├── LICENSE
└── README.md
```

## 快速开始

### 1. 启动后端

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

后端默认运行在：

```text
http://localhost:8000
```

可以打开下面的地址检查服务是否正常：

```text
http://localhost:8000/api/health
```

### 2. 启动前端

另开一个终端：

```powershell
cd frontend
npm install
npm run dev
```

前端默认运行在 Vite 显示的地址，例如：

```text
http://localhost:5173
```

打开前端页面后，选择或拍摄图片，点击“开始检测”即可使用。

## 手机访问

如果想在同一局域网内用手机访问：

1. 确保电脑和手机连接到同一个 Wi-Fi
2. 在电脑上用 `ipconfig` 查看局域网 IP，例如 `192.168.1.23`
3. 手机浏览器打开 Vite 地址，把 `localhost` 换成电脑 IP

示例：

```text
http://192.168.1.23:5173
```

后端需要保持运行，并监听在 `0.0.0.0:8000`。

## 检测模式

### 重复轮廓检测

默认模式。后端会先提取候选色块并描边，再综合比较轮廓、面积、颜色和长宽比，把大量重复出现的相似区域归为同一组。默认只有数量达到 8 个的重复组才会自动计数。

适合：

- 颜色明显、彼此分开的积木或零件
- 轮廓不完全规则，但同批零件形状和大小接近的场景
- 需要排除单个大背景、少量杂物和随机噪点的场景

### 自动纯色检测

后端会在图片中寻找重复出现、颜色相近、大小相对一致的小区域，并返回最可能的目标物体框。

适合：

- 颜色明显的小零件
- 背景较干净的照片
- 物体之间有一定间隔的场景

### 基础轮廓检测

通过灰度、模糊、阈值和轮廓提取识别前景区域。

适合：

- 背景和物体明暗反差明显的照片
- 不依赖颜色的简单计数场景

## API

### 健康检查

```http
GET /api/health
```

响应示例：

```json
{
  "status": "ok"
}
```

### 图片检测

```http
POST /api/detect
```

表单参数：

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `image` | file | 必填 | 要检测的图片 |
| `mode` | string | `repeat_contours` | 检测模式，支持 `repeat_contours`、`auto_color_blocks`、`color`、`basic` |
| `min_area` | number | `900` | 最小区域面积 |
| `min_repeat` | number | `8` | 重复轮廓模式要求的最低相似区域数量 |
| `threshold` | number | `0` | 阈值，`0` 表示自动阈值 |
| `blur` | number | `7` | 模糊半径 |
| `invert` | boolean | `true` | 是否反转前景 |

响应中会返回图片尺寸、检测区域、总数、候选颜色组等信息。

## 使用建议

- 尽量使用光线均匀、背景干净的照片
- 物体之间最好不要严重重叠
- 如果漏检，可以降低“最小面积”
- 如果误检太多，可以提高“最小面积”
- 如果一个框里包含多个物体，可以点击该区域并手动修改数量
- 检测完成后建议人工复核，再导出标注图

## 当前限制

- 项目仍处于开发阶段，功能和界面都可能继续变化
- 不识别物体类别，只负责辅助计数
- 对复杂背景、反光、遮挡和严重重叠场景不稳定
- 自动纯色检测更适合颜色统一的小物体
- 重复轮廓检测要求目标彼此分开，暂不处理严重重叠或粘连
- 目前没有持久化存储，刷新页面后结果不会保留

## 开发命令

前端构建：

```powershell
cd frontend
npm run build
```

前端预览：

```powershell
cd frontend
npm run preview
```

后端开发运行：

```powershell
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

运行后端算法测试：

```powershell
cd backend
python -m unittest discover -s tests -v
```

## 许可证

本项目使用 MIT License，详情见 [LICENSE](./LICENSE)。
