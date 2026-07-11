import { ChangeEvent, MouseEvent, useEffect, useRef, useState } from 'react';

import type { DragEvent as ReactDragEvent, PointerEvent as ReactPointerEvent } from 'react';

type Detection = {
  id: string;
  bbox: [number, number, number, number];
  area: number;
  count: number;
  label?: string;
  manual?: boolean;
};

type ApiDetection = Omit<Detection, 'count'> & { count?: number };

type ResizeCorner = 'nw' | 'ne' | 'sw' | 'se';

type DragState = {
  mode: 'move' | 'resize';
  id: string;
  corner?: ResizeCorner;
  offsetX: number;
  offsetY: number;
  startX: number;
  startY: number;
  originX: number;
  originY: number;
  width: number;
  height: number;
  moved: boolean;
};

type DetectResponse = {
  count: number;
  imageWidth: number;
  imageHeight: number;
  detections: ApiDetection[];
  selectedColor?: AutoColorInfo['selectedColor'] | null;
  selectedScore?: number;
  candidateColorGroups?: AutoColorCandidate[];
  params: {
    mode?: string;
    minArea: number;
    threshold?: number | 'otsu';
    blur?: number;
    invert?: boolean;
    clusterCount?: number;
  };
};

type AutoColorCandidate = {
  index: number;
  color: {
    rgb: [number, number, number] | number[];
    hex: string;
  };
  score: number;
  componentCount: number;
  validCount: number;
  medianArea: number;
  areaCv: number;
  maxAreaRatio: number;
  maskRatio: number;
  edgeRatio: number;
  centerRatio: number;
  slenderRatio: number;
  distanceThreshold: number;
  selected?: boolean;
};

type AutoColorInfo = {
  selectedColor: {
    rgb: [number, number, number] | number[];
    hex: string;
  } | null;
  selectedScore: number;
  candidateColorGroups: AutoColorCandidate[];
};

const defaultMinArea = 900;
const defaultBlur = 7;
const detectionModes = [
  { value: 'auto_color_blocks', label: '自动纯色检测' },
  { value: 'basic', label: '基础轮廓检测' },
] as const;
const resizeCorners: ResizeCorner[] = ['nw', 'ne', 'sw', 'se'];

export function App() {
  const [file, setFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string>('');
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const [detections, setDetections] = useState<Detection[]>([]);
  const [minArea, setMinArea] = useState(defaultMinArea);
  const [threshold, setThreshold] = useState(0);
  const [blur, setBlur] = useState(defaultBlur);
  const [invert, setInvert] = useState(true);
  const [mode, setMode] = useState<(typeof detectionModes)[number]['value']>('auto_color_blocks');
  const [autoColorInfo, setAutoColorInfo] = useState<AutoColorInfo | null>(null);
  const [addMode, setAddMode] = useState(false);
  const [isDetecting, setIsDetecting] = useState(false);
  const [isDraggingFile, setIsDraggingFile] = useState(false);
  const [isMobileReview, setIsMobileReview] = useState(false);
  const [message, setMessage] = useState('请上传一张物体分开、背景尽量干净的照片。');
  const imageRef = useRef<HTMLImageElement | null>(null);
  const dragStateRef = useRef<DragState | null>(null);
  const suppressNextBoxClickRef = useRef(false);

  const totalCount = detections.reduce((sum, detection) => sum + detection.count, 0);

  useEffect(() => {
    const mediaQuery = window.matchMedia('(max-width: 820px)');
    const syncMobileState = () => setIsMobileReview(mediaQuery.matches);

    syncMobileState();
    mediaQuery.addEventListener('change', syncMobileState);
    return () => mediaQuery.removeEventListener('change', syncMobileState);
  }, []);

  useEffect(() => {
    if (isMobileReview) {
      setAddMode(false);
    }
  }, [isMobileReview]);

  function loadImageFile(nextFile: File) {
    if (!nextFile.type.startsWith('image/')) {
      setMessage('请拖入或选择图片文件。');
      return;
    }
    setFile(nextFile);
    setImageUrl((currentUrl) => {
      if (currentUrl) URL.revokeObjectURL(currentUrl);
      return URL.createObjectURL(nextFile);
    });
    setImageSize({ width: 0, height: 0 });
    setDetections([]);
    setAutoColorInfo(null);
    setMessage('照片已载入，可以开始检测。');
  }

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const nextFile = event.target.files?.[0];
    if (!nextFile) return;
    loadImageFile(nextFile);
  }

  function handleImageDrag(event: ReactDragEvent<HTMLElement>) {
    event.preventDefault();
    event.stopPropagation();
    setIsDraggingFile(true);
  }

  function handleImageDragLeave(event: ReactDragEvent<HTMLElement>) {
    event.preventDefault();
    event.stopPropagation();
    if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
      setIsDraggingFile(false);
    }
  }

  function handleImageDrop(event: ReactDragEvent<HTMLElement>) {
    event.preventDefault();
    event.stopPropagation();
    setIsDraggingFile(false);

    const nextFile = Array.from(event.dataTransfer.files).find((item) => item.type.startsWith('image/'));
    if (!nextFile) {
      setMessage('没有找到可用的图片文件。');
      return;
    }
    loadImageFile(nextFile);
  }

  async function runDetection() {
    if (!file) {
      setMessage('请先选择或拍摄一张照片。');
      return;
    }

    const uploadFile = await prepareImageForDetection(file);
    const form = new FormData();
    form.append('image', uploadFile);
    form.append('mode', mode);
    form.append('min_area', String(minArea));
    form.append('threshold', String(threshold));
    form.append('blur', String(blur));
    form.append('invert', String(invert));

    setIsDetecting(true);
    setMessage('正在识别物体，请稍等...');

    try {
      const controller = new AbortController();
      const timeout = window.setTimeout(() => controller.abort(), 45000);
      const apiUrl = getApiUrl('/api/detect');
      let response: Response;
      try {
        response = await fetch(apiUrl, { method: 'POST', body: form, signal: controller.signal });
      } finally {
        window.clearTimeout(timeout);
      }
      if (!response.ok) {
        const error = await response.json().catch(() => null);
        throw new Error(error?.detail ?? '检测失败，请稍后再试。');
      }
      const data = (await response.json()) as DetectResponse;
      const nextDetections = relabel(data.detections.map((detection) => ({ ...detection, count: detection.count ?? 1 })));
      setImageSize({ width: data.imageWidth, height: data.imageHeight });
      setDetections(nextDetections);
      const nextAutoColorInfo =
        data.params.mode === 'auto_color_blocks'
          ? {
              selectedColor: data.selectedColor ?? null,
              selectedScore: data.selectedScore ?? 0,
              candidateColorGroups: data.candidateColorGroups ?? [],
            }
          : null;
      setAutoColorInfo(nextAutoColorInfo);
      const nextTotal = nextDetections.reduce((sum, detection) => sum + detection.count, 0);
      const colorSuffix = nextAutoColorInfo?.selectedColor ? `自动目标色 ${nextAutoColorInfo.selectedColor.hex}。` : '';
      setMessage(`已识别 ${nextDetections.length} 个区域，当前总数为 ${nextTotal}。${colorSuffix}如果一个框里有多个物体，可以手动修改数量。`);
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') {
        setMessage('检测超时。可以换一张更小的照片，或调高最小面积后再试。');
      } else {
        setMessage(error instanceof Error ? error.message : '检测失败，请稍后再试。');
      }
    } finally {
      setIsDetecting(false);
    }
  }

  function removeDetection(id: string) {
    setDetections((current) => relabel(current.filter((item) => item.id !== id)));
  }

  function updateDetectionCount(id: string, count: number) {
    const safeCount = Math.max(1, Math.floor(count || 1));
    setDetections((current) => current.map((item) => (item.id === id ? { ...item, count: safeCount } : item)));
  }

  function moveDetection(id: string, x: number, y: number, width: number, height: number) {
    const nextX = Math.round(clamp(x, 0, Math.max(0, imageSize.width - width)));
    const nextY = Math.round(clamp(y, 0, Math.max(0, imageSize.height - height)));
    setDetections((current) =>
      current.map((item) => (item.id === id ? { ...item, bbox: [nextX, nextY, Math.round(width), Math.round(height)] } : item)),
    );
  }

  function resizeDetection(dragState: DragState, point: { x: number; y: number }) {
    if (!dragState.corner) return;

    const minSize = Math.max(24, Math.min(imageSize.width, imageSize.height) * 0.025);
    let left = dragState.originX;
    let top = dragState.originY;
    let right = dragState.originX + dragState.width;
    let bottom = dragState.originY + dragState.height;

    if (dragState.corner.includes('w')) {
      left = clamp(point.x, 0, right - minSize);
    }
    if (dragState.corner.includes('e')) {
      right = clamp(point.x, left + minSize, imageSize.width);
    }
    if (dragState.corner.includes('n')) {
      top = clamp(point.y, 0, bottom - minSize);
    }
    if (dragState.corner.includes('s')) {
      bottom = clamp(point.y, top + minSize, imageSize.height);
    }

    setDetections((current) =>
      current.map((item) =>
        item.id === dragState.id
          ? {
              ...item,
              bbox: [Math.round(left), Math.round(top), Math.round(right - left), Math.round(bottom - top)],
              area: Math.round((right - left) * (bottom - top)),
            }
          : item,
      ),
    );
  }

  function promptForCount(initialCount: number) {
    const nextValue = window.prompt('这个区域里有几个物体？', String(initialCount));
    if (nextValue === null) return null;
    const parsed = Number(nextValue);
    if (!Number.isFinite(parsed) || parsed < 1) {
      setMessage('数量必须是大于 0 的数字。');
      return null;
    }
    return Math.floor(parsed);
  }

  function setRegionCount(id: string, currentCount: number) {
    const nextCount = promptForCount(currentCount);
    if (nextCount === null) return;
    updateDetectionCount(id, nextCount);
    setMessage(`区域数量已更新为 ${nextCount}。`);
  }

  function getImagePoint(clientX: number, clientY: number) {
    if (!imageRef.current || imageSize.width === 0 || imageSize.height === 0) return null;

    const rect = imageRef.current.getBoundingClientRect();
    return {
      x: ((clientX - rect.left) / rect.width) * imageSize.width,
      y: ((clientY - rect.top) / rect.height) * imageSize.height,
    };
  }

  function handleBoxPointerDown(event: ReactPointerEvent<Element>, detection: Detection) {
    if (isMobileReview) return;

    event.preventDefault();
    event.stopPropagation();

    const point = getImagePoint(event.clientX, event.clientY);
    if (!point) return;

    const [x, y, width, height] = detection.bbox;
    dragStateRef.current = {
      mode: 'move',
      id: detection.id,
      offsetX: point.x - x,
      offsetY: point.y - y,
      startX: point.x,
      startY: point.y,
      originX: x,
      originY: y,
      width,
      height,
      moved: false,
    };
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function handleResizePointerDown(event: ReactPointerEvent<Element>, detection: Detection, corner: ResizeCorner) {
    if (isMobileReview) return;

    event.preventDefault();
    event.stopPropagation();

    const point = getImagePoint(event.clientX, event.clientY);
    if (!point) return;

    const [x, y, width, height] = detection.bbox;
    dragStateRef.current = {
      mode: 'resize',
      id: detection.id,
      corner,
      offsetX: 0,
      offsetY: 0,
      startX: point.x,
      startY: point.y,
      originX: x,
      originY: y,
      width,
      height,
      moved: false,
    };
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function handleBoxPointerMove(event: ReactPointerEvent<Element>) {
    const dragState = dragStateRef.current;
    if (!dragState) return;

    event.preventDefault();
    event.stopPropagation();

    const point = getImagePoint(event.clientX, event.clientY);
    if (!point) return;

    if (Math.abs(point.x - dragState.startX) > 3 || Math.abs(point.y - dragState.startY) > 3) {
      dragState.moved = true;
    }

    if (dragState.mode === 'resize') {
      resizeDetection(dragState, point);
    } else {
      moveDetection(dragState.id, point.x - dragState.offsetX, point.y - dragState.offsetY, dragState.width, dragState.height);
    }
  }

  function handleBoxPointerUp(event: ReactPointerEvent<Element>) {
    const dragState = dragStateRef.current;
    if (!dragState) return;

    event.preventDefault();
    event.stopPropagation();

    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }

    if (dragState.moved) {
      suppressNextBoxClickRef.current = true;
      setMessage(dragState.mode === 'resize' ? '区域大小已调整。' : '区域位置已调整。');
    }
    dragStateRef.current = null;
  }

  function handleBoxClick(event: MouseEvent<Element>, id: string, count: number) {
    event.stopPropagation();
    if (isMobileReview) return;

    if (suppressNextBoxClickRef.current) {
      suppressNextBoxClickRef.current = false;
      return;
    }

    setRegionCount(id, count);
  }

  function handleResizeHandleClick(event: MouseEvent<Element>) {
    event.stopPropagation();
    suppressNextBoxClickRef.current = false;
  }

  function handleImageClick(event: MouseEvent<HTMLDivElement>) {
    if (isMobileReview || !addMode || !imageRef.current || imageSize.width === 0 || imageSize.height === 0) return;

    const nextCount = promptForCount(1);
    if (nextCount === null) return;

    const rect = imageRef.current.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * imageSize.width;
    const y = ((event.clientY - rect.top) / rect.height) * imageSize.height;
    const size = Math.max(Math.min(imageSize.width, imageSize.height) * 0.07, 32);
    const bbox: [number, number, number, number] = [
      Math.max(0, Math.round(x - size / 2)),
      Math.max(0, Math.round(y - size / 2)),
      Math.round(size),
      Math.round(size),
    ];

    setDetections((current) => relabel([...current, { id: `manual_${Date.now()}`, bbox, area: size * size, count: nextCount, manual: true }]));
    setMessage(`已手动添加一个数量为 ${nextCount} 的区域。`);
  }

  function exportAnnotatedImage() {
    if (!imageRef.current || imageSize.width === 0 || imageSize.height === 0) return;

    const canvas = document.createElement('canvas');
    canvas.width = imageSize.width;
    canvas.height = imageSize.height;
    const context = canvas.getContext('2d');
    if (!context) return;

    context.drawImage(imageRef.current, 0, 0, imageSize.width, imageSize.height);
    context.lineWidth = Math.max(3, imageSize.width / 420);
    context.font = `${Math.max(22, imageSize.width / 45)}px sans-serif`;
    context.textBaseline = 'top';

    detections.forEach((detection, index) => {
      const [x, y, w, h] = detection.bbox;
      const text = `${index + 1}${detection.count > 1 ? ` x${detection.count}` : ''}`;
      const labelWidth = Math.max(54, text.length * Math.max(14, imageSize.width / 68));
      context.strokeStyle = detection.manual ? '#d85a2a' : '#2f6f9f';
      context.fillStyle = detection.manual ? '#d85a2a' : '#2f6f9f';
      context.strokeRect(x, y, w, h);
      context.fillRect(x, Math.max(0, y - 32), labelWidth, 30);
      context.fillStyle = '#ffffff';
      context.fillText(text, x + 10, Math.max(0, y - 29));
    });

    const link = document.createElement('a');
    link.download = 'countsnap-annotated.png';
    link.href = canvas.toDataURL('image/png');
    link.click();
  }

  const hasImage = Boolean(imageUrl);
  const scaleX = imageSize.width || 1;
  const scaleY = imageSize.height || 1;
  const imageMeta = imageSize.width && imageSize.height ? `${imageSize.width} x ${imageSize.height}` : '未载入图片';
  const selectedFileName = file?.name ?? '';
  const fileNameParts = file ? splitFileName(file.name) : null;

  return (
    <main className="app-shell">
      <section className="topbar">
        <div className="brand-block">
          <div>
            <p className="eyebrow">CountSnap v0.2</p>
            <h1>拍照数物体</h1>
            <p className="subtitle">上传或现场拍摄照片，自动框选分散物体，再按实际情况微调数量。</p>
          </div>
        </div>
        <div className="count-pill">
          <small>当前总数</small>
          <span>{totalCount}</span>
          <b>{detections.length} 个区域</b>
        </div>
      </section>

      <section className="workspace">
        <aside className="control-panel">
          <div className="panel-heading">
            <span>01</span>
            <div>
              <h2>采集与检测</h2>
              <p>选择照片后运行自动识别。</p>
            </div>
          </div>

          <div
            className={isDraggingFile ? 'upload-box drag-over' : 'upload-box'}
            onDragEnter={handleImageDrag}
            onDragOver={handleImageDrag}
            onDragLeave={handleImageDragLeave}
            onDrop={handleImageDrop}
          >
            <input id="image-upload-input" type="file" accept="image/*" capture="environment" onChange={handleFileChange} />
            <label className="upload-button" htmlFor="image-upload-input">
              选择图片
            </label>
            <strong>选择或拍摄照片</strong>
            {fileNameParts ? (
              <span className="upload-file-name" title={selectedFileName}>
                <span>{fileNameParts.head}</span>
                <b>{fileNameParts.tail}</b>
              </span>
            ) : (
              <span>也可以把图片直接拖到这里或画布区域。</span>
            )}
            {file && <small className="upload-meta">{imageMeta}</small>}
          </div>

          <div className="field-group">
            <div className="wide-field mode-field">
              <span className="field-label">检测模式</span>
              <div className="mode-switch">
                {detectionModes.map((item) => (
                  <button
                    key={item.value}
                    className={mode === item.value ? 'mode-option active' : 'mode-option'}
                    type="button"
                    onClick={() => setMode(item.value)}
                  >
                    {item.label}
                  </button>
                ))}
              </div>
            </div>
            <label className="wide-field native-mode-field">
              检测模式
              <select value={mode} onChange={(event) => setMode(event.target.value as typeof mode)}>
                {detectionModes.map((item) => (
                  <option key={item.value} value={item.value}>
                    {item.label}
                  </option>
                ))}
              </select>
            </label>
            {mode === 'auto_color_blocks' && (
              <div className="wide-field auto-color-card">
                <span>自动目标色</span>
                <div className="auto-color-preview">
                  <i style={{ backgroundColor: autoColorInfo?.selectedColor?.hex ?? '#ffffff' }} />
                  <b>{autoColorInfo?.selectedColor?.hex ?? '等待检测'}</b>
                  {autoColorInfo && <small>评分 {autoColorInfo.selectedScore.toFixed(1)}</small>}
                </div>
                <small>
                  {autoColorInfo
                    ? `${autoColorInfo.candidateColorGroups.length} 个候选颜色组，已按重复小块特征选择。`
                    : '会自动寻找重复出现、大小接近、集中在中间的同色小块。'}
                </small>
              </div>
            )}
            <label>
              最小面积
              <input type="number" min="10" step="50" value={minArea} onChange={(event) => setMinArea(Number(event.target.value))} />
            </label>
            <label className="wide-field">
              阈值
              <input type="number" min="0" max="255" value={threshold} onChange={(event) => setThreshold(Number(event.target.value))} />
              <small>填 0 会使用自动阈值。</small>
            </label>
            <label>
              模糊半径
              <input type="number" min="3" max="31" step="2" value={blur} onChange={(event) => setBlur(Number(event.target.value))} />
            </label>
            <label className="toggle-row">
              <input type="checkbox" checked={invert} onChange={(event) => setInvert(event.target.checked)} />
              反转前景
            </label>
          </div>

          <div className="action-stack">
            <button className="primary-button" disabled={!hasImage || isDetecting} onClick={runDetection}>
              {isDetecting ? '检测中...' : '开始检测'}
            </button>
            <button className={addMode ? 'secondary-button manual-region-button active' : 'secondary-button manual-region-button'} disabled={!hasImage || isMobileReview} onClick={() => setAddMode((value) => !value)}>
              {addMode ? '手动加框已开启' : '手动添加区域'}
            </button>
            <button className="secondary-button" disabled={!detections.length} onClick={exportAnnotatedImage}>
              导出标注图
            </button>
          </div>

          <p className="status-text">{message}</p>
        </aside>

        <section
          className={isDraggingFile ? 'image-stage drag-over' : 'image-stage'}
          onClick={handleImageClick}
          onDragEnter={handleImageDrag}
          onDragOver={handleImageDrag}
          onDragLeave={handleImageDragLeave}
          onDrop={handleImageDrop}
        >
          <div className="stage-header">
            <div>
              <span>02</span>
              <h2>检测画布</h2>
            </div>
            <div className="stage-meta">
              <b>{isMobileReview && hasImage ? '手机端查看标注' : addMode ? '点按图片添加区域' : hasImage ? '拖动框可调整' : '等待照片'}</b>
              <small>{file ? imageMeta : '空画布'}</small>
            </div>
          </div>
          {!hasImage && <div className="empty-state">这里会显示待检测照片。适合苹果、瓶盖、硬币、零件等彼此分开的物体。也可以把图片拖到这里上传。</div>}
          {hasImage && (
            <div className="image-wrap">
              <img
                ref={imageRef}
                src={imageUrl}
                alt="已上传的待检测照片"
                onLoad={(event) => setImageSize({ width: event.currentTarget.naturalWidth, height: event.currentTarget.naturalHeight })}
              />
              <svg className="overlay" viewBox={`0 0 ${scaleX} ${scaleY}`} preserveAspectRatio="none">
                {detections.map((detection, index) => {
                  const [x, y, w, h] = detection.bbox;
                  const displayLabel = `${index + 1}${detection.count > 1 ? ` x${detection.count}` : ''}`;
                  const labelWidth = Math.min(128, scaleX);
                  const labelX = clamp(x, 0, Math.max(0, scaleX - labelWidth));
                  const handleSize = Math.max(14, Math.min(28, Math.min(w, h) * 0.18));
                  const handleOffset = handleSize / 2;
                  const handlePositions: Record<ResizeCorner, { x: number; y: number }> = {
                    nw: { x, y },
                    ne: { x: x + w, y },
                    sw: { x, y: y + h },
                    se: { x: x + w, y: y + h },
                  };
                  return (
                    <g
                      key={detection.id}
                      className={detection.manual ? 'manual-box' : 'detected-box'}
                      onPointerDown={(event) => handleBoxPointerDown(event, detection)}
                      onPointerMove={handleBoxPointerMove}
                      onPointerUp={handleBoxPointerUp}
                      onPointerCancel={handleBoxPointerUp}
                      onClick={(event) => event.stopPropagation()}
                    >
                      <rect
                        className="draggable-box"
                        x={x}
                        y={y}
                        width={w}
                        height={h}
                        rx="6"
                        onClick={(event) => handleBoxClick(event, detection.id, detection.count)}
                      />
                      <foreignObject x={labelX} y={Math.max(0, y - 34)} width={labelWidth} height="32">
                        <button className="box-label" onClick={(event) => handleBoxClick(event, detection.id, detection.count)} title="点击修改数量；拖动框可调整位置">
                          {displayLabel}
                        </button>
                      </foreignObject>
                      {resizeCorners.map((corner) => {
                        const position = handlePositions[corner];
                        return (
                          <rect
                            key={corner}
                            className={`resize-handle resize-${corner}`}
                            x={position.x - handleOffset}
                            y={position.y - handleOffset}
                            width={handleSize}
                            height={handleSize}
                            rx={Math.max(3, handleSize * 0.22)}
                            onPointerDown={(event) => handleResizePointerDown(event, detection, corner)}
                            onClick={handleResizeHandleClick}
                          />
                        );
                      })}
                    </g>
                  );
                })}
              </svg>
            </div>
          )}
        </section>

        <aside className="result-panel">
          <div className="panel-heading result-heading">
            <span>03</span>
            <div>
              <h2>结果复核</h2>
              <p>调整数量、删除误检区域。</p>
            </div>
          </div>
          <div className="result-score">
            <p className="result-number">{totalCount}</p>
            <p className="result-caption">按区域数量汇总后的总数</p>
          </div>
          <div className="count-summary">
            <span>{detections.length}</span>
            <b>个识别区域</b>
          </div>
          <div className="detection-list">
            {!detections.length && <p className="result-empty">{isMobileReview ? '还没有识别区域。上传照片后开始检测。' : '还没有识别区域。上传照片后开始检测，或开启手动加框。'}</p>}
            {detections.map((detection, index) => (
              <div key={detection.id} className="detection-row">
                <button className="detection-main" onClick={() => setRegionCount(detection.id, detection.count)}>
                  <span>#{index + 1}</span>
                  <small>{detection.manual ? '手动添加' : `面积 ${Math.round(detection.area)} px`}</small>
                </button>
                <input
                  aria-label={`第 ${index + 1} 个区域的数量`}
                  type="number"
                  min="1"
                  value={detection.count}
                  onChange={(event) => updateDetectionCount(detection.id, Number(event.target.value))}
                />
                <button className="remove-button" onClick={() => removeDetection(detection.id)} title="移除这个区域">
                  x
                </button>
              </div>
            ))}
          </div>
        </aside>
      </section>
    </main>
  );
}

function getApiUrl(path: string) {
  const { protocol, hostname } = window.location;
  return `${protocol}//${hostname}:8000${path}`;
}

function relabel(items: Detection[]) {
  return items.map((item, index) => ({ ...item, count: Math.max(1, Math.floor(item.count || 1)), label: String(index + 1) }));
}

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

function splitFileName(name: string) {
  if (name.length <= 28) return { head: name, tail: '' };
  return {
    head: name.slice(0, 18),
    tail: name.slice(-12),
  };
}

async function prepareImageForDetection(file: File): Promise<File> {
  const maxSide = 1800;
  if (!file.type.startsWith('image/')) return file;

  const bitmap = await createImageBitmap(file);
  const scale = Math.min(1, maxSide / Math.max(bitmap.width, bitmap.height));
  if (scale >= 1) {
    bitmap.close();
    return file;
  }

  const canvas = document.createElement('canvas');
  canvas.width = Math.round(bitmap.width * scale);
  canvas.height = Math.round(bitmap.height * scale);
  const context = canvas.getContext('2d');
  if (!context) {
    bitmap.close();
    return file;
  }

  context.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  bitmap.close();

  const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.9));
  if (!blob) return file;

  return new File([blob], file.name.replace(/\.[^.]+$/, '') + '-detect.jpg', { type: 'image/jpeg' });
}


