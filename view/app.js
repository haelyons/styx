/* Two views of the same snapshot. Raw records remain available beside every plot. */
const D = JSON.parse(document.getElementById('snapshot').textContent);
const analysis = document.body.dataset.page === 'analysis';
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const numeric = v => typeof v === 'number' && Number.isFinite(v);
const uniq = a => [...new Set(a)].sort((a,b)=>String(a).localeCompare(String(b),undefined,{numeric:true}));
const get = (r,k) => r.fields?.[k];
const num = v => numeric(v) ? v.toLocaleString(undefined,{maximumSignificantDigits:7}) : v === true ? 'yes' : v === false ? 'no' : v ?? 'not recorded';
const average = a => a.length ? a.reduce((s,x)=>s+x,0)/a.length : null;
const median = a => {a=[...a].sort((x,y)=>x-y);return a.length ? (a[Math.floor(a.length/2)]+a[Math.ceil(a.length/2)-1])/2 : null;};
const COLORS=['#2873b8','#c66526','#32836c','#925da3','#ae5369','#6c7b28','#44637b','#b0842c'];
const titles = {'agent_s':'Agent elapsed time (s)','cpu_s':'Recorded CPU time (CPU-s)','output_tokens':'Generated tokens','input_tokens':'Input tokens','cached_tokens':'Cached input tokens','tool_calls':'Tool calls','cost_usd':'Agent inference cost (USD)','total_cost_usd':'Agent inference cost (USD)','usd':'GPU job charges (USD)','gpu_s':'GPU runtime (s)','jobs':'Submitted jobs','failed':'Failed jobs','timeout':'Timed-out jobs','cancelled':'Cancelled jobs','left':'Jobs left running','said.RUNTIME_S':'Predicted runtime (s)','said.PEAK_GPU_MB':'Predicted peak GPU memory (MB)','said.REQUEST_S':'Requested GPU time limit (s)','said.TEST_SCORE':'Predicted test score','seconds':'Measured runtime (s)','limit_s':'Requested limit (s)','peak_gpu_mb':'Peak GPU memory (MB)','peak_rss_mb':'Peak CPU memory (MB)','gpu_busy_share':'GPU busy share','gpu_util_mean':'Mean GPU utilisation (%)','epochs':'Training epochs','test_score':'Test score','test_rmse':'Test RMSE','loss':'Next-token loss','tok_s':'Tokens per second','batch':'Batch size','own.SELF_SECONDS':'Predicted agent elapsed time (s)','own.SELF_OUTPUT_TOKENS':'Predicted generated tokens','own.SELF_TOOL_CALLS':'Predicted tool calls'};
const label = k => titles[k] || k.replace(/^said\.TEST_SCORE_E(\d+)$/, 'Predicted test score at $1 epochs').replaceAll('_',' ');
const metrics = rows => uniq(rows.flatMap(r=>Object.entries(r.fields||{}).filter(([,v])=>numeric(v)).map(([k])=>k)));
const E = new Map(D.episodes.map(e=>[e.id,e]));
const refs = new Map([...D.references,...D.jobs].map(e=>[e.id,e]));
const selection = Object.fromEntries(new URLSearchParams(location.search));
const downloaded = (name,value) => {const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'}));a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);};
$('download').onclick=()=>downloaded('styx-snapshot.json',D);
$('close').onclick=()=>$('detail').close();
function detail(id){
 const r=E.get(id)||refs.get(id);if(!r)return;
 $('detail-content').innerHTML=`<h2>${esc(r.workload)} · ${esc(r.agent)}</h2><p class="muted">${esc(r.id)}</p>${r.url?`<p class="links"><a href="${esc(r.url)}" target="_blank" rel="noopener">Published record</a><a href="${esc(r.transcript)}" target="_blank" rel="noopener">Transcript, if published</a><a href="${esc(r.prompt_url)}" target="_blank" rel="noopener">Exact prompt, if published</a></p>`:''}${r.episode_id?`<button id="parent">Open agent episode</button>`:''}<h3>Task as recorded</h3><pre>${esc(r.prompt||'The exact prompt is not in this local snapshot. Missing traces do not establish that the agent did no computation.')}</pre><h3>Original record</h3><pre>${esc(JSON.stringify(r.raw,null,2))}</pre>`;
 if(r.episode_id)$('parent').onclick=()=>detail(r.episode_id);
 if(!$('detail').open)$('detail').showModal();
}
function control(parent,name,values,value,change){
 const el=document.createElement('label');el.innerHTML=`${esc(name)}<select aria-label="${esc(name)}">${values.map(v=>`<option value="${esc(v[0])}">${esc(v[1])}</option>`).join('')}</select>`;
 parent.append(el);const s=el.querySelector('select');s.value=value;s.onchange=()=>change(s.value);return s;
}
function section(id,title,text){
 const s=document.createElement('section');s.id=id;s.className='section';s.innerHTML=`<h2>${esc(title)}</h2><p>${esc(text)}</p>`;$('content').append(s);return s;
}
function table(parent,rows,cols){
 const div=document.createElement('div');parent.append(div);let page=0,order=null,asc=true;
 const draw=()=>{let a=[...rows];if(order)a.sort((a,b)=>{const x=cols[order].value(a),y=cols[order].value(b);return (numeric(x)&&numeric(y)?x-y:String(x??'').localeCompare(String(y??'')))*(asc?1:-1);});const max=Math.max(1,Math.ceil(a.length/12));page=Math.min(page,max-1);
 div.innerHTML=`<div class="table-scroll"><table><thead><tr>${cols.map((c,i)=>`<th><button data-sort="${i}">${esc(c.label)}${order===i?(asc?' ↑':' ↓'):''}</button></th>`).join('')}</tr></thead><tbody>${a.slice(page*12,page*12+12).map(r=>`<tr ${r.id?`data-id="${esc(r.id)}" tabindex="0"`:''}>${cols.map(c=>`<td>${esc(num(c.value(r)))}</td>`).join('')}</tr>`).join('')}</tbody></table></div><div class="pager"><button data-dir="-1" ${page===0?'disabled':''}>Previous</button><span>${rows.length} rows · page ${page+1} of ${max}</span><button data-dir="1" ${page===max-1?'disabled':''}>Next</button><button data-export>Download rows</button></div>`;
 div.querySelectorAll('[data-sort]').forEach(b=>b.onclick=()=>{const n=Number(b.dataset.sort);asc=n===order?!asc:true;order=n;draw();});
 div.querySelectorAll('[data-dir]').forEach(b=>b.onclick=()=>{page+=Number(b.dataset.dir);draw();});
 div.querySelector('[data-export]').onclick=()=>downloaded('styx-selected-rows.json',rows);
 div.querySelectorAll('[data-id]').forEach(tr=>{tr.onclick=()=>detail(tr.dataset.id);tr.onkeydown=e=>{if(e.key==='Enter')detail(tr.dataset.id);};});};draw();
}
const idcols=[{label:'Agent',value:r=>r.agent},{label:'Workload',value:r=>r.workload},{label:'GPU',value:r=>r.gpu},{label:'Condition',value:r=>r.condition},{label:'Run',value:r=>r.run}];
function plot(parent,rows,x,y,{xl,yl,color=r=>r.agent,diagonal=false,extra=[],log=false}={}){
 const div=document.createElement('div');div.className='plot';parent.append(div);
 const usable=rows.filter(r=>numeric(x(r))&&numeric(y(r))&&(!log||(x(r)>0&&y(r)>0)));
 const p=document.createElement('p');p.className='caption';p.textContent=`${usable.length} of ${rows.length} selected records plotted; ${rows.length-usable.length} missing, non-finite${log?' or non-positive':''}. Click a point for its record. Inclusion rules are stated above; missing values are not zeros.`;parent.append(p);
 if(!usable.length){div.className='empty';div.textContent='No plottable measurements in this selection.';return;}
 const groups=uniq(usable.map(color));const traces=groups.map((g,i)=>{const r=usable.filter(r=>color(r)===g);return {type:'scatter',mode:'markers',name:g,x:r.map(x),y:r.map(y),customdata:r.map(r=>r.id),text:r.map(r=>`${esc(r.agent)} · ${esc(r.workload)}<br>${esc(r.condition)} · ${esc(r.gpu)}<br>${esc(r.run)}`),hovertemplate:'%{text}<br>%{x} / %{y}<extra></extra>',marker:{size:7,opacity:.7,color:COLORS[i%COLORS.length]}};});
 traces.push(...extra);
 if(diagonal){const lo=usable.reduce((v,r)=>Math.min(v,x(r),y(r)),Infinity),hi=usable.reduce((v,r)=>Math.max(v,x(r),y(r)),-Infinity);traces.push({x:[lo,hi],y:[lo,hi],mode:'lines',name:'Equal values',line:{color:'#9ca8b7',dash:'dot'},hoverinfo:'skip'});}
 Plotly.newPlot(div,traces,{height:410,margin:{l:76,r:20,t:20,b:75},paper_bgcolor:'white',plot_bgcolor:'white',font:{family:'system-ui',size:12,color:'#334155'},xaxis:{title:{text:xl},type:log?'log':'linear',gridcolor:'#edf0f4',zerolinecolor:'#cbd5e1'},yaxis:{title:{text:yl},type:log?'log':'linear',gridcolor:'#edf0f4',zerolinecolor:'#cbd5e1'},legend:{orientation:'h',y:-.25},hovermode:'closest'},{responsive:true,displaylogo:false});
 div.on('plotly_click',e=>{const id=e.points[0]?.customdata;if(id)detail(id);});
}
function metricExplorer(parent,rows,keys,initial,{forecast=false,self=false}={}){
 if(!keys.length){parent.insertAdjacentHTML('beforeend','<p class="empty">No numeric fields recorded.</p>');return;}
 const controls=document.createElement('div');controls.className='controls';parent.append(controls);const body=document.createElement('div');parent.append(body);let field=keys.includes(initial)?initial:keys[0];let scale=false;let epochAxis=false;
 const draw=()=>{body.querySelectorAll('.plot').forEach(d=>{if(d.data)Plotly.purge(d);});body.innerHTML='';
 const a=rows.map((r,i)=>({...r,index:i+1}));let extra=[];
 if(forecast){const base=field==='said.RUNTIME_S'?'ours.seconds':field==='said.PEAK_GPU_MB'?'ours.peak_gpu_mb':field==='said.TEST_SCORE'?'ours.test_score':null;
 const r=a.filter(r=>numeric(get(r,field))&&numeric(get(r,field+'_LOW'))&&numeric(get(r,field+'_HIGH'))&&get(r,field+'_LOW')<=get(r,field)&&get(r,field)<=get(r,field+'_HIGH')&&(!scale||get(r,field+'_LOW')>0));
 if(r.length)extra.push({x:r.map(r=>r.index),y:r.map(r=>get(r,field)),mode:'markers',name:'10th–90th percentile',marker:{size:0},error_y:{type:'data',symmetric:false,array:r.map(r=>get(r,field+'_HIGH')-get(r,field)),arrayminus:r.map(r=>get(r,field)-get(r,field+'_LOW')),color:'#8b99aa',thickness:1},hoverinfo:'skip'});
 if(base){const r=a.filter(r=>numeric(get(r,base)));extra.push({x:r.map(r=>r.index),y:r.map(r=>get(r,base)),mode:'markers',name:'Reference at evaluation time',marker:{color:'#17202d',symbol:'line-ew',size:10},customdata:r.map(r=>r.id)});}
 }
 if(self){const r=a.filter(r=>numeric(get(r,field))&&numeric(get(r,field+'_LOW'))&&numeric(get(r,field+'_HIGH'))&&get(r,field+'_LOW')<=get(r,field)&&get(r,field)<=get(r,field+'_HIGH'));const actualKey=({'own.SELF_SECONDS':'agent_s','own.SELF_OUTPUT_TOKENS':'output_tokens','own.SELF_TOOL_CALLS':'tool_calls'})[field];if(actualKey)extra.push({x:r.map(r=>get(r,actualKey)),y:r.map(r=>get(r,field)),mode:'markers',name:'10th–90th percentile',marker:{size:0},error_y:{type:'data',symmetric:false,array:r.map(r=>get(r,field+'_HIGH')-get(r,field)),arrayminus:r.map(r=>get(r,field)-get(r,field+'_LOW')),color:'#8b99aa'},hoverinfo:'skip'});}
 const actual=self?({'own.SELF_SECONDS':'agent_s','own.SELF_OUTPUT_TOKENS':'output_tokens','own.SELF_TOOL_CALLS':'tool_calls'})[field]:null;
 plot(body,a,r=>actual?get(r,actual):epochAxis?get(r,'epochs'):r.index,r=>get(r,field),{xl:actual?label(actual):epochAxis?'Training epochs':'Record order (stable within selection)',yl:label(field)+(field.includes('TEST_SCORE')&&selection.workload&&D.workloads?.[selection.workload]?.metric?' ('+D.workloads[selection.workload].metric+')':''),extra,diagonal:!!actual,log:scale,color:r=>r.kind?r.gpu:r.agent});
 if(forecast&&field.includes('TEST_SCORE')&&uniq(rows.map(r=>r.workload)).length>1)body.insertAdjacentHTML('afterbegin','<p class="notice">Different workloads may use different metrics. Select one workload before comparing score magnitudes.</p>');
 table(body,rows,[...idcols,...(rows.some(r=>r.kind)?[{label:'Variant',value:r=>get(r,'variant')},{label:'Seed',value:r=>get(r,'seed')},{label:'Epochs',value:r=>get(r,'epochs')},{label:'Exit code',value:r=>get(r,'exit')}]:[]),{label:label(field),value:r=>get(r,field)},...(forecast?[{label:'10th percentile',value:r=>get(r,field+'_LOW')},{label:'90th percentile',value:r=>get(r,field+'_HIGH')}]:[]),{label:'Status',value:r=>r.status??get(r,'state')??(get(r,'timeout')===true?'timed out':numeric(get(r,'exit'))?'exit '+get(r,'exit'):'not recorded')}]);};
 control(controls,'Measurement',keys.map(k=>[k,label(k)]),field,v=>{field=v;draw();});control(controls,'Axes',[['linear','Linear (includes zero)'],['log','Logarithmic (positive values only)']],'linear',v=>{scale=v==='log';draw();});if(rows.some(r=>r.kind==='training'))control(controls,'Horizontal axis',[['order','Record order'],['epochs','Training epochs']],'order',v=>{epochAxis=v==='epochs';draw();});draw();
}
function intro(){
 const m=D.meta;$('intro').innerHTML=`<p class="eyebrow">Compute access · agent behaviour</p><h1>${analysis?'What changes when agents can compute?':'The experiment, as recorded'}</h1><p class="lead">${analysis?'Hypotheses about taking opportunities, learning from computation, and choosing GPU limits. Each plot links back to the observations.':'What we asked, which options were available, what agents reported, and what the executions measured. Individual records come before comparisons.'}</p><p class="muted">Built ${esc(m.built)} · Data snapshot ${esc(m.published||'time not recorded')} · ${esc(m.revision?m.revision.slice(0,12):'data revision not recorded')} · view code ${esc(m.code.slice(0,12))}</p><p class="notice">This is a frozen export, not a live run monitor. The episode source contains scored records; absent records do not establish that an attempt was never started. ${m.revision?'Published record links use the pinned data revision.':'Published links use the moving main revision; downloaded records and input hashes preserve this export.'}</p><details><summary>Snapshot provenance and unavailable inputs</summary><pre>${esc(JSON.stringify(m,null,2))}</pre></details>`;
 const dims=['agent','workload','gpu','condition','run'];dims.forEach(k=>{const values=uniq([...D.episodes,...(['workload','gpu'].includes(k)?D.references:[])].map(e=>e[k]));control($('filters'),k==='agent'?'Agent (CLI + model)':k==='gpu'?'Target GPU':k[0].toUpperCase()+k.slice(1),[['', 'All'],...values.map(v=>[v,v])],selection[k]||'',v=>{selection[k]=v;render();});});const b=document.createElement('button');b.textContent='Reset';b.onclick=()=>{location.href=location.pathname;};$('filters').append(b);const nav=document.createElement('nav');nav.id='sections';nav.className='jump';nav.setAttribute('aria-label','Sections');$('filters').after(nav);
}
function filtered(){return D.episodes.filter(e=>['agent','workload','gpu','condition','run'].every(k=>!selection[k]||e[k]===selection[k]));}
function results(e){
 let s=section('tasks','1. Tasks and available choices','A workload is a computation. An episode is an agent’s attempt. A submitted GPU job is one execution within that attempt. Reference executions are measurements made by the experimenters.');
 s.insertAdjacentHTML('beforeend',`<div class="table-scroll"><table><thead><tr><th>Task</th><th>Held fixed within a comparison</th><th>Varied by the experiment</th><th>Agent chooses</th></tr></thead><tbody><tr><td>Forecast a training run</td><td>Target script, data and environment, except the factor compared</td><td>Agent, target GPU, CPU access and proxy instruction</td><td>Whether and what to compute; forecasts and requested limit</td></tr><tr><td>Measure next-token loss</td><td>Specified inference workload within a cell</td><td>Agent, workload and price of unused reserved time</td><td>GPU, commands, requested limits, retries and cancellation</td></tr></tbody></table></div><p>Read only: inspect files. CPU available: local execution is permitted. CPU + proxy instruction: the same resources, with an instruction to find and use a local proxy. These are assigned conditions, not a classification of observed actions.</p><p>Training epochs, the agent’s time allowance, and the requested GPU time limit are separate quantities. Prompts and prices changed during collection. Open an episode for its exact prompt when available; current definitions are not applied retrospectively.</p><p class="links"><a href="https://github.com/haelyons/styx/blob/${D.meta.code}/tasks/forecast.py">Forecast definition at view revision</a><a href="https://github.com/haelyons/styx/blob/${D.meta.code}/tasks/post.py">Execution definition at view revision</a></p>`);
 const catalog=document.createElement('details');catalog.innerHTML='<summary>Training workload catalogue at this code revision</summary>';s.append(catalog);table(catalog,Object.entries(D.workloads||{}).map(([key,v])=>({key,...v})),[{label:'ID',value:r=>r.key},{label:'Task',value:r=>r.task},{label:'Metric',value:r=>r.metric},{label:'Source script',value:r=>r.script}]);
 s=section('coverage','2. Recorded episodes','Every selected episode appears below, including errors, timeouts and missing answers. A status of “recorded” means the record exists, not that the answer is correct.');
 const counts={};e.forEach(r=>{counts[r.status]=(counts[r.status]||0)+1;});s.insertAdjacentHTML('beforeend',`<div class="pills">${Object.entries(counts).map(([k,v])=>`<span class="pill">${esc(k)}: ${v}</span>`).join('')}</div>`);
 table(s,e,[...idcols,{label:'Task',value:r=>r.task},{label:'Status',value:r=>r.status},{label:'Answered',value:r=>get(r,'answered')},{label:'Correct',value:r=>get(r,'correct')}]);
 const rr=D.references.filter(r=>(!selection.workload||r.workload===selection.workload)&&(!selection.gpu||r.gpu===selection.gpu));
 s=section('references','3. Reference executions','Individual measurements, including failed executions. Reference filters use workload and GPU only; agent, condition and run filters do not apply. The input file and original row are retained.');
 const controlBox=document.createElement('div');controlBox.className='controls';s.append(controlBox);const refBody=document.createElement('div');s.append(refBody);
 const refDraw=kind=>{refBody.innerHTML='';const rows=rr.filter(r=>r.kind===kind);metricExplorer(refBody,rows,metrics(rows),'seconds');};control(controlBox,'Reference workload type',[['training','Training'],['inference','Inference']],'training',refDraw);refDraw('training');
 s=section('activity','4. Agent activity and expenditure','Elapsed time, CPU time, tokens, calls and charges are distinct measurements. CPU time includes process overhead and historical metering differs. It does not establish whether a useful experiment ran. Transcripts are linked from each episode; no computation-use classification is inferred from these counters.');
 const activity=metrics(e).filter(k=>!k.includes('.')&&!/(_err|_in|_pin|_base|_recalled|_ratio|_ln|request_|runtime_|memory_|score_|self_|^epoch$|^exit$|^hold$|^said$)/.test(k));
 metricExplorer(s,e,activity,'cpu_s');
 const fc=e.filter(r=>r.task==='forecast');s=section('forecasts','5. Forecasts and requested GPU limits','Reported numbers in their original units. Bars show valid reported 10th–90th percentiles; black marks use the reference saved with that episode. These are separate from the latest reference executions above. Missing values remain in the table. No prediction is inferred for an unasked field.');
 metricExplorer(s,fc,metrics(fc).filter(k=>k.startsWith('said.')&&!/_(LOW|HIGH)$/.test(k)&&!k.includes('TEST_RMSE')),'said.RUNTIME_S',{forecast:true});
 const po=e.filter(r=>r.task==='post'),ids=new Set(po.map(r=>r.id)),jj=D.jobs.filter(r=>ids.has(r.episode_id));
 s=section('jobs','6. Submitted GPU jobs and reported loss','One row per available submitted job, retaining submission order, states and commands in its record. An episode with no submitted job is distinct from one whose job failed. If only a flattened episode is available, only its first job can be shown.');
 s.insertAdjacentHTML('beforeend',`<p class="caption">${jj.length} job records from ${po.length} execution episodes; ${jj.filter(j=>!j.complete).length} are first-job-only extracts. Full job histories are not reconstructed from aggregate counters.</p>`);
 metricExplorer(s,jj,metrics(jj),'seconds');
 table(s,po,[...idcols,{label:'Reported loss',value:r=>r.raw.said},{label:'Reference loss',value:r=>get(r,'ours.loss')},{label:'Correct',value:r=>get(r,'correct')},{label:'Jobs',value:r=>get(r,'jobs')},{label:'GPU charges (USD)',value:r=>get(r,'usd')},{label:'Unused-time price share',value:r=>get(r,'hold')}]);
 s=section('self','7. Forecasts of the agent’s own work','Asked in a separate session before the task: predicted elapsed time, generated tokens and tool calls. These are not pre-computation forecasts of the target training job. Equality is shown as a dotted line.');
 metricExplorer(s,e,metrics(e).filter(k=>k.startsWith('own.')&&!/_(LOW|HIGH)$/.test(k)),'own.SELF_SECONDS',{self:true});
 s=section('inventory','8. Complete field inventory','All scalar fields in selected records are listed, including diagnostics and legacy names. Select any numeric field to plot it. Original nested arrays and complete records are retained in the snapshot download and record inspector. This inventory makes omissions visible rather than claiming every scalar is a distinct scientific result.');
 const all=uniq(e.flatMap(r=>Object.keys(r.fields)));table(s,all.map(k=>({key:k,present:e.filter(r=>get(r,k)!==null&&get(r,k)!==undefined).length,numeric:e.filter(r=>numeric(get(r,k))).length})),[{label:'Field',value:r=>r.key},{label:'Recorded',value:r=>r.present},{label:'Numeric',value:r=>r.numeric},{label:'Missing',value:r=>e.length-r.present}]);
 const d=document.createElement('details');d.innerHTML='<summary>Plot any recorded numeric field</summary>';s.append(d);metricExplorer(d,e,metrics(e),'cpu_s');
}
const err=(r,p,t)=>{const a=get(r,p),b=get(r,t);return numeric(a)&&numeric(b)&&a>0&&b>0?Math.abs(Math.log(a/b)):null;};
const usable=r=>r.status==='recorded'&&get(r,'peeked')!==true;
function matched(rows,before,after){
 const blocks=new Map();
 for(const r of rows.filter(r=>r.task==='forecast')){
  const key=JSON.stringify([r.run,r.agent,r.workload,r.gpu,get(r,'host')??'unknown',get(r,'cli')??'unknown',get(r,'effort')??'unknown']);
  if(!blocks.has(key))blocks.set(key,{});const b=blocks.get(key);(b[r.condition]??=[]).push(r);
 }
 const out=[];
 for(const b of blocks.values()){
  if(!b[before]||!b[after])continue;
  const a=b[before],z=b[after],template=z[0];
  const mean=(rs,fn)=>average(rs.filter(usable).map(fn).filter(numeric));
  const diff=fn=>{const p=mean(a,fn),q=mean(z,fn);return numeric(p)&&numeric(q)?q-p:null;};
  out.push({...template,id:template.id,before:a,after:z,runtimeGain:diff(r=>{const v=err(r,'said.RUNTIME_S','ours.seconds');return numeric(v)?-v:null;}),scoreGain:diff(r=>{const v=err(r,'said.TEST_SCORE','ours.test_score');return numeric(v)?-v:null;}),requestChange:diff(r=>get(r,'said.REQUEST_S')),forecastChange:diff(r=>get(r,'said.RUNTIME_S')),cpuChange:diff(r=>get(r,'cpu_s')),elapsedChange:diff(r=>get(r,'agent_s')),beforeN:a.length,afterN:z.length});
 }
 return out;
}
function analyses(e){
 let s=section('questions','Questions guiding this page','The hypotheses are that agents may leave available compute unused; use it without obtaining relevant evidence; obtain evidence without changing a forecast; or change a forecast without changing allocation. An unchanged answer can also be appropriate. These plots describe associations and condition differences, not internal causes.');
 s.insertAdjacentHTML('beforeend','<p class="notice">Direct / local / proxy describe assigned access and instructions. CPU time and tool calls are not verified measures of useful computation. Confirmed non-use and evidence quality require the trace; they remain unknown where the trace has not been inspected.</p>');
 const f=e.filter(r=>r.task==='forecast');
 s=section('access','1. Does offering compute change the result?','Within each run, agent, workload, target GPU, host, CLI version and effort, repeats are averaged inside each condition. All matched blocks remain in the table; error plots require usable positive forecasts and references. Errors/timeouts and answer-source flags are excluded only from error averages. Unknown metadata weakens matching. These are separate episodes, not before-and-after beliefs.');
 const controls=document.createElement('div');controls.className='controls';s.append(controls);const body=document.createElement('div');s.append(body);
 function drawPair(choice){body.querySelectorAll('.plot').forEach(d=>{if(d.data)Plotly.purge(d);});body.innerHTML='';const [a,b]=choice.split(':'),rows=matched(f,a,b);
 plot(body,rows,r=>r.forecastChange,r=>r.requestChange,{xl:'Change in predicted runtime (s)',yl:'Change in requested GPU limit (s)'});
 plot(body,rows,r=>r.runtimeGain,r=>r.scoreGain,{xl:'Runtime error reduction (positive = closer)',yl:'Score error reduction (positive = closer)'});
 table(body,rows,[...idcols,{label:'Earlier-condition records',value:r=>r.beforeN},{label:'Later-condition records',value:r=>r.afterN},{label:'Runtime error reduction',value:r=>r.runtimeGain},{label:'Score error reduction',value:r=>r.scoreGain},{label:'CPU change (s)',value:r=>r.cpuChange},{label:'Elapsed change (s)',value:r=>r.elapsedChange},{label:'Forecast change (s)',value:r=>r.forecastChange},{label:'Requested limit change (s)',value:r=>r.requestChange}]);
 body.insertAdjacentHTML('beforeend','<p class="caption">Error is |ln(prediction / saved reference)|. A block is not an independent workload. A point opens one later-condition episode; the row download retains all episodes from both conditions.</p>');}
 control(controls,'Condition contrast',[['direct:local','Read only → CPU available'],['local:proxy','CPU available → proxy instruction']],'direct:local',drawPair);drawPair('direct:local');
 s=section('compute','2. Does recorded computation accompany better forecasts?','CPU time against target prediction error. Non-users and failures remain in the population; only missing numeric values prevent a point. No causal benefit of computation is inferred: agents select whether and how to compute, and CPU meters differ across historical runs.');
 plot(s,f,r=>get(r,'cpu_s'),r=>err(r,'said.RUNTIME_S','ours.seconds'),{xl:'Recorded CPU time (CPU-s; zero retained)',yl:'Absolute log runtime error',color:r=>r.condition});
 s=section('allocation','3. Do requested limits follow forecasts?','The reported runtime forecast and requested GPU time limit are separate decisions. The equality line shows a request equal to the median forecast. Rounding, uncertainty and pricing can all influence allocation. Older prompts did not specify the current unused-time charge.');
 plot(s,f,r=>get(r,'said.RUNTIME_S'),r=>get(r,'said.REQUEST_S'),{xl:'Predicted runtime (s)',yl:'Requested GPU time limit (s)',color:r=>r.condition,diagonal:true});
 const ratio=f.map(r=>({...r,ratio:numeric(get(r,'ours.seconds'))&&get(r,'ours.seconds')>0&&numeric(get(r,'said.RUNTIME_S'))?get(r,'said.RUNTIME_S')/get(r,'ours.seconds'):null,margin:numeric(get(r,'said.RUNTIME_S'))&&get(r,'said.RUNTIME_S')>0&&numeric(get(r,'said.REQUEST_S'))?get(r,'said.REQUEST_S')/get(r,'said.RUNTIME_S'):null}));
 const dd=document.createElement('details');dd.innerHTML='<summary>Forecast error and requested multiplier</summary><p>The axes share the runtime forecast. With a fixed requested limit, lowering the forecast mechanically raises the multiplier. This is not independent evidence of greater caution. Neither padding nor this multiplier is labelled a safety margin without trace evidence.</p>';s.append(dd);plot(dd,ratio,r=>r.ratio,r=>r.margin,{xl:'Predicted / reference runtime',yl:'Requested limit / predicted runtime',color:r=>r.condition,log:true});
 s=section('pricing','4. Does allocation respond to the price of unused time?','Execution episodes with a recorded price share. Earlier missing prices remain unknown, not zero. Compare one workload and agent at a time; unbalanced run versions and GPU choices can affect the comparison.');
 const po=e.filter(r=>r.task==='post'),withPrice=po.filter(r=>numeric(get(r,'hold')));
 const priceRows=withPrice.map(r=>{const first=D.jobs.find(j=>j.episode_id===r.id&&j.order===1);return {...r,firstLimit:first?get(first,'limit_s'):null};});
 plot(s,priceRows,r=>get(r,'hold'),r=>r.firstLimit,{xl:'Price of unused second / price of running second',yl:'First requested GPU limit (s)',color:r=>r.workload});
 s.insertAdjacentHTML('beforeend',`<p class="caption">${po.length-withPrice.length} execution episodes have no recorded price share. A first-job limit is not total episode expenditure; all submitted jobs are on Results.</p>`);
 s=section('uncertainty','5. Are intervals informative and calibrated?','Coverage against the individual reference measurements saved with each episode, where present. If only its reference summary was saved, coverage below is only inclusion of that summary and is labelled separately. Wider intervals can increase coverage without better point forecasts.');
 const covered=f.map(r=>{const lo=get(r,'said.RUNTIME_S_LOW'),hi=get(r,'said.RUNTIME_S_HIGH');const values=r.raw.seeds?.seconds;const ys=Array.isArray(values)?values.filter(numeric):numeric(get(r,'ours.seconds'))?[get(r,'ours.seconds')]:[];const valid=numeric(lo)&&numeric(hi)&&lo<=hi&&ys.length;return {...r,width:valid?hi-lo:null,coverage:valid?average(ys.map(t=>Number(t>=lo&&t<=hi))):null,coverageKind:Array.isArray(values)?'Saved reference executions':'Saved summary only'};});
 plot(s,covered,r=>r.width,r=>r.coverage,{xl:'Reported runtime interval width (s)',yl:'Reference coverage (0–1)',color:r=>r.coverageKind});
 s=section('self-analysis','6. Does self-forecasting track workload forecasting?','Absolute log errors of predicted agent elapsed time and target runtime. Association can reflect workload, model or harness differences. A self-forecast is not a pre-probe estimate of the target job.');
 plot(s,f,r=>err(r,'own.SELF_SECONDS','agent_s'),r=>err(r,'said.RUNTIME_S','ours.seconds'),{xl:'Absolute log error: own elapsed time',yl:'Absolute log error: target runtime'});
 s=section('basis','Interpretation and denominators','The complete selected population is retained on Results. Matching, missing observations, exclusions and different reference versions limit the comparisons here. There is no aggregate capability score or model leaderboard.');
 table(s,e,[...idcols,{label:'Status',value:r=>r.status},{label:'Answer-source flag',value:r=>get(r,'peeked')},{label:'Recorded CPU seconds',value:r=>get(r,'cpu_s')},{label:'Runtime error',value:r=>err(r,'said.RUNTIME_S','ours.seconds')},{label:'Score error',value:r=>err(r,'said.TEST_SCORE','ours.test_score')}]);
}
function render(){
 const q=new URLSearchParams(Object.entries(selection).filter(([,v])=>v));history.replaceState(null,'',location.pathname+(q.size?'?'+q:''));document.querySelectorAll('header nav a').forEach(a=>a.href=(a.textContent==='Results'?'index.html':'analysis.html')+(q.size?'?'+q:''));
 $('content').querySelectorAll('.plot').forEach(d=>{if(d.data)Plotly.purge(d);});$('content').innerHTML='';
 const e=filtered();$('counts').textContent=`${e.length} of ${D.episodes.length} recorded episodes selected · ${uniq(e.map(r=>r.run)).length} run identifiers · ${uniq(e.map(r=>r.workload)).length} workloads`;
 if(analysis)analyses(e);else results(e);$('sections').innerHTML=[...document.querySelectorAll('#content section')].map(s=>`<a href="#${s.id}">${esc(s.querySelector('h2').textContent.replace(/^\d+\. /,''))}</a>`).join('');
}
intro();render();
