"use strict";
const $=id=>document.getElementById(id);
const S={rotation:0,meta:null,version:0,line:null,lineId:null,start:null,end:null,image:null,imageRect:null,drag:null,dirty:true,seq:0,job:null,history:new Map()};
let previewTimer=null,mapSeq=0;
function status(text,error=false){$('status').textContent=text;$('status').classList.toggle('error',error);}
async function api(path,body){const opt=body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json','X-STM-Token':window.STM_TOKEN},body:JSON.stringify(body)};const response=await fetch(path,opt);const value=await response.json();if(!response.ok)throw Error(value.error||'请求失败');return value;}
function fmt(v,n=3){return Number(Number(v).toFixed(n)).toString();}
function value(id){return Number($(id).value);}
function canvasContext(id){const c=$(id),r=c.getBoundingClientRect(),ratio=window.devicePixelRatio||1;if(c.width!==Math.round(r.width*ratio)||c.height!==Math.round(r.height*ratio)){c.width=Math.round(r.width*ratio);c.height=Math.round(r.height*ratio);}const ctx=c.getContext('2d');ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,r.width,r.height);ctx.font='11px Microsoft YaHei, sans-serif';return {ctx,w:r.width,h:r.height};}
function placeholder(id,text){const {ctx,w,h}=canvasContext(id);ctx.fillStyle='#8a999e';ctx.textAlign='center';ctx.fillText(text,w/2,h/2);}
function setDirty(){S.dirty=true;updateButtons();}
function updateButtons(){document.querySelectorAll('[data-export]').forEach(b=>b.disabled=!S.line||S.dirty||!!S.job);$('reverseLine').disabled=!S.end;$('newLine').disabled=!S.meta;}
function settings(){return {rotation_quarters:S.rotation,start:S.start,end:S.end,curves:value('curveCount'),normalization_mode:$('normalizationMode').value,normalize:$('normalizationMode').value!=='none',norm_lo_mV:value('normLo'),norm_hi_mV:value('normHi'),zero_anchor:$('zeroAnchor').checked,offset:value('offset'),energy_min_mV:value('energyMin'),energy_max_mV:value('energyMax'),view:$('mapView').value,map_bias_mV:currentMapBias(),topo_plane:$('topoPlane').checked,profile_mode:$('profileMode').value,heat_min_percentile:value('heatMin'),heat_max_percentile:value('heatMax'),fft_window:$('fftWindow').value,smoothing_method:$('smoothingMethod').value,smoothing_order:value('smoothingOrder'),smoothing_sigma:value('smoothingSigma'),x_offset_mV:value('xOffset'),x_factor:value('xFactor'),y_factor:value('yFactor'),spectrum_min_mV:value('procMin'),spectrum_max_mV:value('procMax'),current_reference_sample:value('currentReference')-1};}
function updateProcessingControls(){
    const mode=$('normalizationMode').value,smooth=$('smoothingMethod').value;
    $('legacyNormFields').hidden=mode!=='abs_mean';$('referenceField').hidden=mode!=='current';
    $('yFactor').disabled=['minmax','first','abs_mean'].includes(mode);
    const descriptions={none:'g × y_factor；不改变各谱相对幅度。',minmax:'裁切后逐谱 [g−min(g)]/[max(g)−min(g)]；原程序忽略 y_factor。',first:'裁切后逐谱除以第一个保留值；原程序忽略 y_factor。',current:`gᵢ × y_factor × I参考/Iᵢ；电流取首个采集偏压 ${S.meta?fmt(S.meta.current_first_bias_mV):'—'} mV，参考谱序号从 A 起。`,abs_mean:'兼容旧记录：逐谱除以 |V| 窗口均值。'};
    $('normalizationHelp').textContent=descriptions[mode];
    $('smoothingOrder').disabled=smooth==='none';$('smoothingSigma').disabled=smooth!=='gaussian';
    const order=value('smoothingOrder');$('smoothingHelp').textContent=smooth==='none'?'不平滑；保留全部实测能量点。':`窗口 ${2*order+1} 点；裁掉两端各 ${order} 点。${smooth==='adjacent'?'近邻均值按原程序从左到右顺序更新。':`σ=${value('smoothingSigma')} 个偏压点。`}`;
}
function currentMapBias(){return S.meta?S.meta.bias_mV[value('mapBias')]:0;}
function setCoords(){if(!S.start||!S.end)return;const p=[...toView(S.start),...toView(S.end)];['ax','ay','bx','by'].forEach((id,i)=>$(id).value=p[i].toFixed(0));}
function restoreConfig(cfg){
    S.rotation=cfg.rotation_quarters||0;updateRotation();S.start=[...cfg.start];S.end=[...cfg.end];setCoords();
    const map={curveCount:'curves',normLo:'norm_lo_mV',normHi:'norm_hi_mV',offset:'offset',energyMin:'energy_min_mV',energyMax:'energy_max_mV',xOffset:'x_offset_mV',xFactor:'x_factor',yFactor:'y_factor',smoothingOrder:'smoothing_order',smoothingSigma:'smoothing_sigma'};
    const defaults={x_offset_mV:0,x_factor:1,y_factor:1,smoothing_order:3,smoothing_sigma:1};Object.entries(map).forEach(([id,k])=>$(id).value=cfg[k]??defaults[k]??$(id).value);
    $('normalizationMode').value=cfg.normalization_mode||(cfg.normalize?'abs_mean':'none');$('smoothingMethod').value=cfg.smoothing_method||'none';$('currentReference').value=(cfg.current_reference_sample??0)+1;
    $('procMin').value=cfg.spectrum_min_mV??S.meta.bias_mV[0];$('procMax').value=cfg.spectrum_max_mV??S.meta.bias_mV.at(-1);
    $('zeroAnchor').checked=cfg.zero_anchor;$('mapView').value=cfg.view;$('topoPlane').checked=cfg.topo_plane;$('profileMode').value=cfg.profile_mode||'raw';$('fftWindow').value=cfg.fft_window||'hann';$('heatMin').value=cfg.heat_min_percentile??1;$('heatMax').value=cfg.heat_max_percentile??99;
    const k=S.meta.bias_mV.reduce((best,v,i)=>Math.abs(v-cfg.map_bias_mV)<Math.abs(S.meta.bias_mV[best]-cfg.map_bias_mV)?i:best,0);$('mapBias').value=k;updateProcessingControls();refreshMapControls();loadMap();
}
function updateHistory(){const select=$('lineHistory');select.replaceChildren();if(!S.history.size){const o=new Option('尚未画线','');select.add(o);}for(const [id,r]of S.history)select.add(new Option(`${id} · ${r.length_nm.toFixed(3)} nm`,id));select.value=S.lineId||'';}
async function installState(state){S.rotation=0;S.meta=state.dataset;updateRotation();S.version=state.data_version;S.history.clear();S.lineId=null;S.line=null;S.start=null;S.end=null;S.dirty=true;S.seq++;$('profileMode').value='raw';$('fftWindow').value='hann';$('filePath').value=S.meta?.source_path||state.default_file||'';if(!S.meta){status('先打开一个 Nanonis 3DS，然后在左图拖出直线。');redrawAll();return;}const m=S.meta;$('normalizationMode').querySelector('[value="current"]').disabled=!m.current_available;if(!m.current_available&&$('normalizationMode').value==='current')$('normalizationMode').value='none';$('procMin').value=Math.max(-30,m.bias_mV[0]);$('procMax').value=Math.min(30,m.bias_mV.at(-1));if(value('procMin')>=value('procMax')){$('procMin').value=m.bias_mV[0];$('procMax').value=m.bias_mV.at(-1);}$('currentReference').value=1;updateProcessingControls();$('fileInfo').textContent=`${m.filename} · ${m.nx} × ${m.ny} 像素 · ${m.n_bias} 个偏压（${fmt(m.bias_mV[0])} → ${fmt(m.bias_mV.at(-1))} mV）`;$('mapBias').max=m.bias_mV.length-1;$('mapBias').value=m.bias_mV.reduce((best,v,i)=>Math.abs(v)<Math.abs(m.bias_mV[best])?i:best,0);$('energyMin').value=Math.max(-40,m.bias_mV[0]);$('energyMax').value=Math.min(40,m.bias_mV.at(-1));if(value('energyMin')>=value('energyMax')){$('energyMin').value=m.bias_mV[0];$('energyMax').value=m.bias_mV.at(-1);}const outer=Math.max(...m.bias_mV.map(Math.abs));$('normLo').value=+(outer*.75).toFixed(4);$('normHi').value=outer;['ax','bx'].forEach(id=>$(id).max=m.nx-1);['ay','by'].forEach(id=>$(id).max=m.ny-1);for(const id of ['ax','ay','bx','by'])$(id).value='';for(const item of state.lines)S.history.set(item.line_id,item);updateHistory();refreshMapControls();await loadMap();updateButtons();status('已载入。在 Topo 或 mapping 上按住鼠标拖出 A→B 直线。');redrawAll();}
function refreshMapControls(){const mapping=$('mapView').value==='mapping';$('biasControl').hidden=!mapping;$('topoPlane').parentElement.hidden=mapping;$('mapBiasLabel').textContent=`${fmt(currentMapBias())} mV`;}
async function loadMap(){if(!S.meta)return;const seq=++mapSeq;const img=new Image();img.src=`/api/map?view=${$('mapView').value}&bias=${currentMapBias()}&plane=${$('topoPlane').checked}&v=${S.version}`;await new Promise((ok,no)=>{img.onload=ok;img.onerror=()=>no(Error('图像载入失败。'));});if(seq!==mapSeq)return;S.image=img;drawMap();}
function viewSize(){return S.rotation%2?[S.meta.ny,S.meta.nx]:[S.meta.nx,S.meta.ny];}
function toView(p){const [x,y]=p,n=S.meta.nx,m=S.meta.ny;return [[x,y],[m-1-y,x],[n-1-x,m-1-y],[y,n-1-x]][S.rotation];}
function fromView(p){const [x,y]=p,n=S.meta.nx,m=S.meta.ny;return [[x,y],[y,m-1-x],[n-1-x,m-1-y],[n-1-y,x]][S.rotation];}
function updateRotation(){if(!S.meta)return;const [nx,ny]=viewSize();['ax','bx'].forEach(id=>$(id).max=nx-1);['ay','by'].forEach(id=>$(id).max=ny-1);$('rotationLabel').textContent=`${S.rotation*90}°`;}
function rotateMap(step){if(!S.meta||S.drag)return;S.rotation=(S.rotation+step+4)%4;updateRotation();setCoords();drawMap();if(S.start&&S.end)schedulePreview(0);}
function mapPoint(xy){const r=S.imageRect,[nx,ny]=viewSize(),p=toView(xy);return [r.x+(p[0]+.5)*r.w/nx,r.y+(p[1]+.5)*r.h/ny];}
function pointFromEvent(event){const c=$('mapCanvas').getBoundingClientRect(),r=S.imageRect,[nx,ny]=viewSize();return fromView([Math.max(0,Math.min(nx-1,Math.round((event.clientX-c.left-r.x)/r.w*nx-.5))),Math.max(0,Math.min(ny-1,Math.round((event.clientY-c.top-r.y)/r.h*ny-.5)))]);}
function insideMap(event){const c=$('mapCanvas').getBoundingClientRect(),r=S.imageRect,x=event.clientX-c.left,y=event.clientY-c.top;return x>=r.x&&x<=r.x+r.w&&y>=r.y&&y<=r.y+r.h;}
function drawMap(){if(!S.meta||!S.image){placeholder('mapCanvas','打开 3DS 后显示 Topo');return;}const {ctx,w,h}=canvasContext('mapCanvas');const [nx,ny]=viewSize();const scale=Math.min((w-62)/nx,(h-52)/ny);const r={x:42+(w-62-nx*scale)/2,y:12+(h-52-ny*scale)/2,w:nx*scale,h:ny*scale};S.imageRect=r;ctx.imageSmoothingEnabled=false;ctx.save();ctx.translate(r.x+r.w/2,r.y+r.h/2);ctx.rotate(S.rotation*Math.PI/2);ctx.drawImage(S.image,-S.meta.nx*scale/2,-S.meta.ny*scale/2,S.meta.nx*scale,S.meta.ny*scale);ctx.restore();ctx.strokeStyle='#d4dde0';ctx.lineWidth=1;ctx.strokeRect(r.x,r.y,r.w,r.h);ctx.fillStyle='#73838a';ctx.font='10px Microsoft YaHei';ctx.textAlign='center';for(let j=0;j<=4;j++){const x=(nx-1)*j/4;ctx.fillText(fmt(x,0),r.x+(x+.5)*scale,r.y+r.h+16);}ctx.textAlign='right';for(let j=0;j<=4;j++){const y=(ny-1)*j/4;ctx.fillText(fmt(y,0),r.x-7,r.y+(y+.5)*scale+3);}ctx.textAlign='center';ctx.fillText('显示 x / px',r.x+r.w/2,h-4);if(S.start&&S.end){const a=mapPoint(S.start),b=mapPoint(S.end);ctx.save();ctx.beginPath();ctx.rect(r.x,r.y,r.w,r.h);ctx.clip();ctx.strokeStyle='#0bffff';ctx.lineWidth=1.8;ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();ctx.restore();[a,b].forEach((p,i)=>{ctx.beginPath();ctx.arc(...p,6,0,2*Math.PI);ctx.fillStyle='#12e0db';ctx.fill();ctx.strokeStyle='#fff';ctx.lineWidth=1.5;ctx.stroke();ctx.font='bold 12px Microsoft YaHei';ctx.textAlign='left';ctx.fillStyle='#00bab6';ctx.fillText(i===0?'A':'B',p[0]+9,p[1]-7);});}}
function axisPlot(ctx,r,xmin,xmax,ymin,ymax,xTitle,yTitle){ctx.strokeStyle='#c6d1d4';ctx.lineWidth=1;ctx.strokeRect(r.x,r.y,r.w,r.h);ctx.fillStyle='#6b7d85';ctx.font='10px Microsoft YaHei';for(let i=0;i<=4;i++){const x=r.x+r.w*i/4,y=r.y+r.h*i/4;ctx.textAlign='center';ctx.fillText(fmt(xmin+(xmax-xmin)*i/4,2),x,r.y+r.h+17);ctx.textAlign='right';ctx.fillText(fmt(ymax-(ymax-ymin)*i/4,2),r.x-7,y+3);}ctx.textAlign='center';ctx.font='11px Microsoft YaHei';ctx.fillText(xTitle,r.x+r.w/2,r.y+r.h+36);ctx.save();ctx.translate(14,r.y+r.h/2);ctx.rotate(-Math.PI/2);ctx.fillText(yTitle,0,0);ctx.restore();}
function quantile(values,p){const a=[...values].sort((x,y)=>x-y);const z=p*(a.length-1),i=Math.floor(z);return a[i]+(a[Math.min(i+1,a.length-1)]-a[i])*(z-i);}
function energyIndices(){const c=S.line.config;return S.line.bias_mV.map((v,i)=>v>=c.energy_min_mV-1e-7&&v<=c.energy_max_mV+1e-7?i:-1).filter(i=>i>=0);}
function nearestPosition(s){return S.line.s_nm.reduce((best,v,i)=>Math.abs(v-s)<Math.abs(S.line.s_nm[best]-s)?i:best,0);}
function drawZ(){
    const mapping=$('mapView').value==='mapping';$('profileTitle').textContent=mapping?'沿线 dI/dV 变化 · LI X':'沿线 Z 高度变化';
    if(!$('profilePanel').open){S.zRect=null;return;}
    if(!S.line){placeholder('zCanvas','画线后显示当前通道的沿线剖面');S.zRect=null;return;}
    const {ctx,w,h}=canvasContext('zCanvas'),l=S.line,r={x:70,y:20,w:w-96,h:h-70},z=l.profile_values;
    // Display a stated axis offset only when raw absolute Z would make ticks unreadable.
    const reference=l.profile_unit==='pm'&&Math.abs(z[0])>10000?Math.trunc(z[0]/1000)*1000:0;
    const zz=z.map(v=>v-reference);let ymin=Math.min(...zz),ymax=Math.max(...zz),span=Math.max(ymax-ymin,.001);ymin-=.08*span;ymax+=.08*span;
    const X=v=>r.x+v/l.length_nm*r.w,Y=v=>r.y+(ymax-v)/(ymax-ymin)*r.h;
    ctx.save();ctx.beginPath();ctx.rect(r.x,r.y,r.w,r.h);ctx.clip();ctx.strokeStyle='#a56b2a';ctx.lineWidth=1.4;ctx.beginPath();
    zz.forEach((v,i)=>{if(i===0)ctx.moveTo(X(l.s_nm[i]),Y(v));else ctx.lineTo(X(l.s_nm[i]),Y(v));});ctx.stroke();ctx.restore();
    axisPlot(ctx,r,0,l.length_nm,ymin,ymax,'沿 A→B 距离 / nm',`${l.profile_channel}${reference?' − 轴偏置':''} / ${l.profile_unit}`);S.zRect=r;
    const mode={raw:'Raw · 原始值',subtract_average:'Subtract average · 减沿线均值',linear_fit:'Linear fit · 去沿线线性背景'}[l.config.profile_mode];
    $('zProfileNote').textContent=`${mode}${l.profile_bias_mV===null?'':` · ${fmt(l.profile_bias_mV)} mV`}`;
    if(reference){ctx.fillStyle='#6b7d85';ctx.font='10px Microsoft YaHei';ctx.textAlign='left';ctx.fillText(`轴偏置 +${reference} pm（原始值保留）`,r.x,12);}
    $('zTip').textContent=`${l.n_samples} 个实测像素 · ${l.profile_unit} · 悬停读取像素、原始值及处理值。`;
}
function fftReadout(f,index){return `f=${fmt(f.frequency_cycles_per_nm[index],5)} cycles/nm · 周期 λ=${fmt(f.period_nm[index],5)} nm · 幅度 ${fmt(f.amplitude[index],6)} ${f.unit}`;}
function drawFFT(){
    S.fftRect=null;
    const f=S.line?.fft;
    $('fftTitle').textContent=f?`沿线一维 FFT · ${f.channel}`:'沿线一维 FFT';
    if(!$('fftPanel').open)return;
    if(!f){placeholder('fftCanvas','画线后预览当前剖面的一维 FFT');$('fftSummary').textContent='画线后显示空间频率与幅度；悬停读取周期。';return;}
    if(!f.available){placeholder('fftCanvas',f.reason);$('fftSourceNote').textContent=f.reason;$('fftSummary').textContent='请拉长线段。';$('fftSampling').textContent='';return;}
    const mode={raw:'Raw',subtract_average:'减均值',linear_fit:'去线性背景'}[f.profile_mode];
    $('fftSourceNote').textContent=`${f.channel}${f.bias_mV===null?'':` · ${fmt(f.bias_mV)} mV`} · ${mode} · FFT 减均值 · ${f.window==='hann'?'Hann':'无窗'}`;
    const {ctx,w,h}=canvasContext('fftCanvas'),r={x:60,y:15,w:w-86,h:h-68},freq=f.frequency_cycles_per_nm,amp=f.amplitude;
    let ymax=Math.max(...amp.slice(1))*1.12;if(ymax<=0)ymax=1;
    const exponent=ymax<.01?Math.floor(Math.log10(ymax)):0,scale=10**exponent;
    const X=v=>r.x+v/f.nyquist_cycles_per_nm*r.w,Y=v=>r.y+r.h-v/ymax*r.h;
    ctx.save();ctx.beginPath();ctx.rect(r.x,r.y,r.w,r.h);ctx.clip();ctx.strokeStyle='#087e82';ctx.lineWidth=1.4;ctx.beginPath();
    for(let i=1;i<freq.length;i++){if(i===1)ctx.moveTo(X(freq[i]),Y(amp[i]));else ctx.lineTo(X(freq[i]),Y(amp[i]));}ctx.stroke();
    if(f.peak){ctx.beginPath();ctx.arc(X(f.peak.frequency_cycles_per_nm),Y(f.peak.amplitude),3,0,2*Math.PI);ctx.fillStyle='#087e82';ctx.fill();}ctx.restore();
    axisPlot(ctx,r,0,f.nyquist_cycles_per_nm,0,ymax/scale,'空间频率 f / cycles·nm⁻¹',`幅度 / ${f.unit}${exponent?` (×10^${exponent})`:''}`);
    S.fftRect=r;
    $('fftSummary').textContent=f.peak?`最大幅值频点：${fftReadout(f,f.peak.index)}`:'去均值后无可辨幅值；悬停读取频点。';
    $('fftSampling').textContent=`${f.resampled?'FFT 专用等距重采样；插值可能减弱高频幅度':'等距像素剖面直接 FFT'} · N=${f.sample_count} · Δf=${fmt(f.frequency_spacing_cycles_per_nm,5)} cycles/nm · 不补零`;
}

function drawHeat(){
    $('heatMinLabel').textContent=`P${fmt(value('heatMin'),1)}`;$('heatMaxLabel').textContent=`P${fmt(value('heatMax'),1)}`;
    if(!S.line){placeholder('heatCanvas','画线后预览能量—位置图');return;}
    const {ctx,w,h}=canvasContext('heatCanvas'),r={x:52,y:15,w:w-105,h:h-60},l=S.line,c=l.config,ks=energyIndices(),E=ks.map(k=>l.bias_mV[k]);
    const all=l.signal_pA.flatMap(row=>ks.map(k=>row[k])),lo=quantile(all,value('heatMin')/100),hi=Math.max(quantile(all,value('heatMax')/100),lo+1e-12),span=hi-lo;
    const edges=l.position_edges_nm,edgeE=E.map((v,i)=>i===0?v-(E[1]-v)/2:(E[i-1]+v)/2);edgeE.push(E.at(-1)+(E.at(-1)-E.at(-2))/2);
    const X=v=>r.x+v/l.length_nm*r.w,Y=v=>r.y+(c.energy_max_mV-v)/(c.energy_max_mV-c.energy_min_mV)*r.h;
    ctx.save();ctx.beginPath();ctx.rect(r.x,r.y,r.w,r.h);ctx.clip();
    for(let j=0;j<ks.length;j++)for(let i=0;i<l.n_samples;i++){
        const v=l.signal_pA[i][ks[j]],rgb=S.meta.palette[Math.round(Math.max(0,Math.min(1,(v-lo)/span))*255)];ctx.fillStyle=`rgb(${rgb})`;
        ctx.fillRect(X(edges[i]),Y(edgeE[j+1]),X(edges[i+1])-X(edges[i])+.3,Y(edgeE[j])-Y(edgeE[j+1])+.3);
    }ctx.restore();axisPlot(ctx,r,0,l.length_nm,c.energy_min_mV,c.energy_max_mV,'沿 A→B 投影距离 / nm','偏压 / mV');
    for(let j=0;j<r.h;j++){const rgb=S.meta.palette[Math.round((1-j/r.h)*255)];ctx.fillStyle=`rgb(${rgb})`;ctx.fillRect(r.x+r.w+10,r.y+j,9,1.2);}
    ctx.font='9px Microsoft YaHei';ctx.textAlign='left';ctx.fillStyle='#6c7d84';ctx.fillText(fmt(hi,3),r.x+r.w+21,r.y+7);ctx.fillText(fmt(lo,3),r.x+r.w+21,r.y+r.h);S.heatRect=r;
    $('heatMinLabel').textContent=`P${fmt(value('heatMin'),1)} · ${fmt(lo)} pA`;$('heatMaxLabel').textContent=`P${fmt(value('heatMax'),1)} · ${fmt(hi)} pA`;
}
function drawWaterfall(){
    if(!S.line){$('distanceLegend').hidden=true;placeholder('waterfallCanvas','画线后预览瀑布曲线');return;}
    const {ctx,w,h}=canvasContext('waterfallCanvas'),r={x:60,y:16,w:w-114,h:h-64},l=S.line,c=l.config,bias=l.waterfall_bias_mV||l.bias_mV;
    const ks=bias.map((v,i)=>v>=c.energy_min_mV-1e-7&&v<=c.energy_max_mV+1e-7?i:-1).filter(i=>i>=0),all=l.waterfall.flatMap(row=>ks.map(k=>row[k]));
    let ymin=Math.min(...all),ymax=Math.max(...all),span=Math.max(ymax-ymin,1e-8);ymin-=.05*span;ymax+=.08*span;
    const X=e=>r.x+(e-c.energy_min_mV)/(c.energy_max_mV-c.energy_min_mV)*r.w,Y=v=>r.y+(ymax-v)/(ymax-ymin)*r.h;
    ctx.save();ctx.beginPath();ctx.rect(r.x,r.y,r.w,r.h);ctx.clip();ctx.lineWidth=1;
    for(let j=0;j<l.waterfall.length;j++){ctx.strokeStyle=l.waterfall_colors_hex[j];ctx.beginPath();ks.forEach((k,n)=>{const x=X(bias[k]),y=Y(l.waterfall[j][k]);if(n===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);});ctx.stroke();}ctx.restore();
    ctx.font='8px Microsoft YaHei';ctx.textAlign='left';const rightK=ks.reduce((best,k)=>bias[k]>bias[best]?k:best,ks[0]);for(let j=0;j<l.waterfall.length;j++){ctx.fillStyle=l.waterfall_colors_hex[j];ctx.fillText(l.sample_names[l.waterfall_indices[j]].split('_').at(-1),r.x+r.w+4,Y(l.waterfall[j][rightK])+2);}
    const p=l.processing||{label:c.normalize?'旧版外侧归一化':'不归一化',unit:c.normalize?'a.u.':'pA',smoothing:{method:'none'}};
    axisPlot(ctx,r,c.energy_min_mV,c.energy_max_mV,ymin,ymax,'处理后偏压 / mV',`LI X / ${p.unit} + offset`);$('waterfallUnit').textContent=p.unit;
    const sm={none:'不平滑',adjacent:'近邻均值',gaussian:'高斯平滑'}[p.smoothing.method];$('waterfallNote').textContent=`${p.label} · ${sm}${p.smoothing.method==='none'?'':` ${p.smoothing.window_points} 点`} · ${c.zero_anchor?`${fmt(l.zero_anchor_bias_mV)} mV 对齐`:'保留基线'}`;
    $('distanceLegend').hidden=false;$('distanceRamp').style.background=`linear-gradient(to right,${S.meta.distance_palette.join(',')})`;$('distanceA').textContent='A · 0 nm';$('distanceB').textContent=`B · ${fmt(l.length_nm)} nm`;
    $('waterfallTip').textContent=`展示 ${l.waterfall.length}/${l.n_samples} 条 · 处理后 ${bias.length} 个偏压点 · 间距 ${c.offset}；原始完整谱另存。`;
}
function redrawAll(){if(S.start&&S.end)$('mapTip').textContent=`像素 A (${S.start.join(', ')}) → B (${S.end.join(', ')}) · 方向键平移 1 px`;if(S.line){$('lineInfo').textContent=`${S.line.line_id} · ${S.line.length_nm.toFixed(4)} nm · ${S.line.n_samples} 个实测像素 × ${S.line.bias_mV.length} 个偏压`;}drawMap();drawZ();drawFFT();drawHeat();drawWaterfall();updateButtons();}
async function requestPreview(){if(!S.meta||!S.start||!S.end)return;const seq=++S.seq;setDirty();status('正在生成线谱预览…');try{const line=await api('/api/line',{settings:settings(),line_id:S.lineId,data_version:S.version});if(seq!==S.seq)return;S.line=line;S.start=[...line.config.start];S.end=[...line.config.end];setCoords();S.lineId=line.line_id;S.history.set(line.line_id,line);S.dirty=false;updateHistory();$('lineInfo').textContent=`${line.line_id} · ${line.length_nm.toFixed(4)} nm · ${line.n_samples} 个实测像素 × ${line.bias_mV.length} 个偏压`;$('mapTip').textContent=`像素 A (${line.config.start.join(', ')}) → B (${line.config.end.join(', ')}) · 方向键整体平移 1 px`; status('预览已更新。端点为整数像素，方向键整体平移 1 px。');redrawAll();}catch(ex){if(seq===S.seq){status(ex.message,true);updateButtons();}}}
function schedulePreview(delay=300){S.seq++;setDirty();clearTimeout(previewTimer);previewTimer=setTimeout(requestPreview,delay);}
function newLine(){clearTimeout(previewTimer);S.seq++;S.lineId=null;S.line=null;S.start=null;S.end=null;S.dirty=true;for(const id of ['ax','ay','bx','by'])$(id).value='';$('lineInfo').textContent='在左图按住鼠标拖出新线。';$('lineHistory').value='';status('新建线：按住拖出 A→B。');redrawAll();}
function lineGrip(pointer){
    if(!S.start||!S.end)return null;
    const a=mapPoint(S.start),b=mapPoint(S.end);
    if(Math.hypot(pointer[0]-a[0],pointer[1]-a[1])<13)return 'start';
    if(Math.hypot(pointer[0]-b[0],pointer[1]-b[1])<13)return 'end';
    const dx=b[0]-a[0],dy=b[1]-a[1],den=dx*dx+dy*dy;
    if(den===0)return null;
    const t=((pointer[0]-a[0])*dx+(pointer[1]-a[1])*dy)/den;
    if(t>=0&&t<=1&&Math.hypot(pointer[0]-a[0]-t*dx,pointer[1]-a[1]-t*dy)<=9)return 'line';
    return null;
}
$('mapCanvas').addEventListener('pointerdown',e=>{
    if(!S.meta||!S.imageRect||!insideMap(e))return;
    e.preventDefault();$('mapCanvas').focus({preventScroll:true});clearTimeout(previewTimer);S.seq++;const c=$('mapCanvas').getBoundingClientRect(),p=[e.clientX-c.left,e.clientY-c.top],grip=lineGrip(p);
    if(grip){S.drag=grip;if(grip==='line')S.dragOrigin={pointer:pointFromEvent(e),start:[...S.start],end:[...S.end]};}
    else{S.drag='end';S.start=pointFromEvent(e);S.end=[...S.start];S.lineId=null;S.line=null;}
    S.dirty=true;$('mapCanvas').setPointerCapture(e.pointerId);setCoords();redrawAll();
});
$('mapCanvas').addEventListener('pointermove',e=>{
    if(!S.meta||!S.imageRect)return;
    const p=pointFromEvent(e);
    if(S.drag){
        if(S.drag==='line'){
            const o=S.dragOrigin,delta=p.map((v,i)=>Math.round(v-o.pointer[i]));
            for(let i=0;i<2;i++)delta[i]=Math.max(-Math.min(o.start[i],o.end[i]),Math.min((i===0?S.meta.nx:S.meta.ny)-1-Math.max(o.start[i],o.end[i]),delta[i]));
            S.start=o.start.map((v,i)=>v+delta[i]);S.end=o.end.map((v,i)=>v+delta[i]);
        }else S[S.drag]=p;
        setCoords();drawMap();
    }else{
        const c=$('mapCanvas').getBoundingClientRect(),grip=lineGrip([e.clientX-c.left,e.clientY-c.top]);
        $('mapCanvas').style.cursor=grip==='line'?'move':grip?'grab':'crosshair';
        $('mapTip').textContent=grip==='line'?'拖动线段中部，平移整条线；长度和方向保持不变。':`显示像素 (${toView(p).join(', ')}) · 源像素 (${p.join(', ')}) · 拖动端点 / 整条线；方向键平移 1 px。`;
    }
});
function pointerEnd(e){if(!S.drag)return;S.drag=null;S.dragOrigin=null;if($('mapCanvas').hasPointerCapture(e.pointerId))$('mapCanvas').releasePointerCapture(e.pointerId);schedulePreview(0);}
$('mapCanvas').addEventListener('pointerup',pointerEnd);$('mapCanvas').addEventListener('pointercancel',pointerEnd);
$('profilePanel').addEventListener('toggle',drawZ);
$('fftPanel').addEventListener('toggle',drawFFT);
$('zCanvas').addEventListener('pointermove',e=>{
    if(!S.line||!S.zRect)return;const b=$('zCanvas').getBoundingClientRect(),r=S.zRect,x=e.clientX-b.left,y=e.clientY-b.top;if(x<r.x||x>r.x+r.w||y<r.y||y>r.y+r.h)return;
    const l=S.line,index=nearestPosition((x-r.x)/r.w*l.length_nm);
    $('zTip').textContent=`${l.sample_names[index]} · 像素 (${l.xy[index].join(', ')}) · 原始 ${fmt(l.profile_raw[index],4)} ${l.profile_unit} · 处理后 ${fmt(l.profile_values[index],4)} ${l.profile_unit}`;
});
$('fftCanvas').addEventListener('pointermove',e=>{
    if(!S.fftRect||!S.line?.fft?.available)return;
    const b=$('fftCanvas').getBoundingClientRect(),r=S.fftRect,x=e.clientX-b.left,y=e.clientY-b.top;
    if(x<r.x||x>r.x+r.w||y<r.y||y>r.y+r.h)return;
    const f=S.line.fft,q=(x-r.x)/r.w*f.nyquist_cycles_per_nm,index=Math.max(1,Math.min(f.frequency_cycles_per_nm.length-1,Math.round(q/f.frequency_spacing_cycles_per_nm)));
    $('fftSummary').textContent=fftReadout(f,index);
});
$('heatCanvas').addEventListener('pointermove',e=>{
    if(!S.line||!S.heatRect)return;const b=$('heatCanvas').getBoundingClientRect(),r=S.heatRect,x=e.clientX-b.left,y=e.clientY-b.top;if(x<r.x||x>r.x+r.w||y<r.y||y>r.y+r.h)return;
    const l=S.line,c=l.config,index=nearestPosition((x-r.x)/r.w*l.length_nm),E=c.energy_max_mV-(y-r.y)/r.h*(c.energy_max_mV-c.energy_min_mV),k=l.bias_mV.reduce((best,v,i)=>Math.abs(v-E)<Math.abs(l.bias_mV[best]-E)?i:best,0);
    $('heatTip').textContent=`${l.sample_names[index]} · 像素 (${l.xy[index].join(', ')}) · s=${fmt(l.s_nm[index])} nm · ${fmt(l.bias_mV[k])} mV · ${fmt(l.signal_pA[index][k],5)} pA`;
});
document.addEventListener('keydown',e=>{
    const moves={ArrowLeft:[-1,0],ArrowRight:[1,0],ArrowUp:[0,-1],ArrowDown:[0,1]},d=moves[e.key];
    if(!d||!S.start||!S.end||S.drag||e.altKey||e.ctrlKey||e.metaKey||e.target.closest('input,select,textarea,button,[contenteditable="true"]'))return;
    e.preventDefault();const origin=fromView([0,0]),delta=fromView(d).map((v,i)=>v-origin[i]);const points=[S.start,S.end].map(p=>p.map((v,i)=>v+delta[i]));
    if(points.some(p=>p[0]<0||p[0]>=S.meta.nx||p[1]<0||p[1]>=S.meta.ny)){status('已到图像边界，整条线保持原位。');return;}
    [S.start,S.end]=points;setCoords();drawMap();schedulePreview(100);
});
for(const id of ['ax','ay','bx','by'])$(id).addEventListener('input',()=>{if(['ax','ay','bx','by'].some(k=>$(k).value===''))return;const [nx,ny]=viewSize(),clamp=(v,n)=>Math.max(0,Math.min(n-1,Math.round(v)));S.start=fromView([clamp(value('ax'),nx),clamp(value('ay'),ny)]);S.end=fromView([clamp(value('bx'),nx),clamp(value('by'),ny)]);setCoords();drawMap();schedulePreview();});
for(const id of ['curveCount','normalizationMode','normLo','normHi','zeroAnchor','offset','energyMin','energyMax','profileMode','fftWindow','smoothingMethod','smoothingOrder','smoothingSigma','xOffset','xFactor','yFactor','procMin','procMax','currentReference'])$(id).addEventListener('input',()=>{if(id==='normalizationMode')$('offset').value=['minmax','first','abs_mean'].includes($('normalizationMode').value)?.024:.5;updateProcessingControls();drawMap();schedulePreview();});
for(const id of ['heatMin','heatMax'])$(id).addEventListener('input',()=>{if(value('heatMin')>=value('heatMax'))$(id).value=id==='heatMin'?value('heatMax')-.5:value('heatMin')+.5;drawHeat();schedulePreview();});
for(const id of ['mapView','mapBias','topoPlane'])$(id).addEventListener('input',async()=>{if(S.line)setDirty();refreshMapControls();try{await loadMap();if(S.line)schedulePreview();}catch(ex){status(ex.message,true);}});
$('rotateCW').onclick=()=>rotateMap(1);$('rotateCCW').onclick=()=>rotateMap(-1);$('newLine').onclick=newLine;$('reverseLine').onclick=()=>{[S.start,S.end]=[S.end,S.start];setCoords();drawMap();schedulePreview(0);};
$('lineHistory').onchange=async()=>{clearTimeout(previewTimer);const seq=++S.seq;const id=$('lineHistory').value;if(!id)return;try{const line=await api('/api/line?id='+encodeURIComponent(id));if(seq!==S.seq)return;S.line=line;S.lineId=id;S.dirty=false;restoreConfig(line.config);updateHistory();$('lineInfo').textContent=`${id} · ${line.length_nm.toFixed(4)} nm · ${line.n_samples} 个实测像素`;redrawAll();status('已恢复当前会话中的线段。');}catch(ex){status(ex.message,true);}};
$('loadPath').onclick=async()=>{try{status('正在读取 3DS…');await installState(await api('/api/load',{path:$('filePath').value}));}catch(ex){status(ex.message,true);}};
$('chooseFile').onclick=()=>$('fileUpload').click();$('fileUpload').onchange=async()=>{const file=$('fileUpload').files[0];if(!file)return;try{status('正在将文件复制到本地工具并载入…');const res=await fetch('/api/upload',{method:'POST',headers:{'X-STM-Token':window.STM_TOKEN,'X-Filename':encodeURIComponent(file.name)},body:file}),data=await res.json();if(!res.ok)throw Error(data.error);await installState(data);}catch(ex){status(ex.message,true);}};
$('restoreLine').onclick=async()=>{try{const line=await api('/api/restore',{path:$('restorePath').value});S.line=line;S.start=[...line.config.start];S.end=[...line.config.end];setCoords();S.lineId=line.line_id;S.history.set(line.line_id,line);S.dirty=false;restoreConfig(line.config);updateHistory();redrawAll();status('已核对源文件，恢复端点与画线参数。');}catch(ex){status(ex.message,true);}};
function showExport(job){$('exportResult').hidden=false;$('exportFolder').textContent=job.folder;$('exportLinks').replaceChildren();for(const f of job.files){if(!/\.(png|svg|pdf|opju|csv|json)$/.test(f.name)||f.name==='Origin_validation.json')continue;const a=document.createElement('a');a.href=f.url;a.textContent=f.name;a.download=f.name;$('exportLinks').append(a);}}
async function pollJob(id){try{const job=await api('/api/job?id='+id);status(job.message,job.status==='error');if(job.status==='running'){setTimeout(()=>pollJob(id),700);return;}S.job=null;updateButtons();if(job.status==='complete')showExport(job);}catch(ex){S.job=null;status(ex.message,true);updateButtons();}}
document.querySelectorAll('[data-export]').forEach(b=>b.onclick=async()=>{if(!S.line||S.dirty)return;try{S.job='pending';updateButtons();const job=await api('/api/export',{line_id:S.lineId,kind:b.dataset.export});S.job=job.job_id;status(job.message);pollJob(job.job_id);}catch(ex){S.job=null;status(ex.message,true);updateButtons();}});
$('stopServer').onclick=async()=>{try{const r=await api('/api/shutdown',{});status(r.message);document.querySelectorAll('button').forEach(b=>b.disabled=true);}catch(ex){status(ex.message,true);}};
// Resizing changes only CSS dimensions; sampling coordinates/data are unchanged.
let resizeFrame=null;
const layoutKey='stm-workbench-layout-v1';
function saveLayout(){try{const layout={widths:{},heights:{}};document.querySelectorAll('.plot-window').forEach(c=>{if(c.style.flexBasis)layout.widths[c.id]=c.style.flexBasis;});document.querySelectorAll('.plotframe').forEach(f=>{if(f.style.height)layout.heights[f.dataset.canvas]=f.style.height;});localStorage.setItem(layoutKey,JSON.stringify(layout));}catch{}}
function loadLayout(){try{const l=JSON.parse(localStorage.getItem(layoutKey)||'{}');Object.entries(l.widths||{}).forEach(([id,v])=>{if($(id))$(id).style.flexBasis=v;});Object.entries(l.heights||{}).forEach(([id,v])=>{if($(id))$(id).parentElement.style.height=v;});}catch{}}
function applyPlotSize(card,frame,w,h,axis){if(axis!=='y')card.style.flexBasis=Math.max(280,Math.min(document.querySelector('.plots').clientWidth,w))+'px';if(axis!=='x')frame.style.height=Math.max(frame.classList.contains('smallframe')?150:180,Math.min(1400,h))+'px';}
document.querySelectorAll('.resize-handle').forEach(handle=>{
    let grip=null;const frame=handle.closest('.plotframe'),card=handle.closest('.plot-window'),axis=handle.dataset.axis;
    handle.addEventListener('pointerdown',e=>{e.preventDefault();e.stopPropagation();grip={x:e.clientX,y:e.clientY,w:card.getBoundingClientRect().width,h:frame.getBoundingClientRect().height};handle.setPointerCapture(e.pointerId);document.body.classList.add('resizing');});
    handle.addEventListener('pointermove',e=>{if(!grip)return;applyPlotSize(card,frame,grip.w+e.clientX-grip.x,grip.h+e.clientY-grip.y,axis);});
    const end=e=>{if(!grip)return;grip=null;document.body.classList.remove('resizing');if(handle.hasPointerCapture(e.pointerId))handle.releasePointerCapture(e.pointerId);saveLayout();};
    handle.addEventListener('pointerup',end);handle.addEventListener('pointercancel',end);
    handle.addEventListener('keydown',e=>{const d={ArrowLeft:[-20,0],ArrowRight:[20,0],ArrowUp:[0,-20],ArrowDown:[0,20]}[e.key];if(!d)return;e.preventDefault();e.stopPropagation();applyPlotSize(card,frame,card.getBoundingClientRect().width+d[0],frame.getBoundingClientRect().height+d[1],axis);saveLayout();});
});
$('resetLayout').onclick=()=>{document.querySelectorAll('.plot-window').forEach(c=>c.style.removeProperty('flex-basis'));document.querySelectorAll('.plotframe').forEach(f=>f.style.removeProperty('height'));try{localStorage.removeItem(layoutKey);}catch{}redrawAll();};
loadLayout();updateProcessingControls();
const redrawObserver=new ResizeObserver(()=>{if(resizeFrame!==null)return;resizeFrame=requestAnimationFrame(()=>{resizeFrame=null;redrawAll();});});
document.querySelectorAll('.plotframe').forEach(frame=>redrawObserver.observe(frame));
api('/api/state').then(async state=>{const installSeq=S.seq+1;await installState(state);if(S.seq!==installSeq)return;if(S.meta&&state.lines.length){const seq=++S.seq,id=state.lines.at(-1).line_id,line=await api('/api/line?id='+encodeURIComponent(id));if(seq!==S.seq)return;S.line=line;S.lineId=id;S.history.set(id,line);S.dirty=false;restoreConfig(line.config);updateHistory();$('lineInfo').textContent=`${id} · ${line.length_nm.toFixed(4)} nm · ${line.n_samples} 个实测像素`;redrawAll();status('已恢复当前会话的最后一条线，可继续微调或导出。');}}).catch(ex=>status('无法连接本地服务：'+ex.message,true));
