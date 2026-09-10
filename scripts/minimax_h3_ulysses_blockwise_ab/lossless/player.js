const vs=[...document.querySelectorAll('video')];let playing=false;
function seek(t){vs.forEach(v=>v.currentTime=t)}
document.querySelector('#play').onclick=async()=>{seek(vs[0].currentTime);vs[0].muted=false;vs[1].muted=true;try{await Promise.all(vs.map(v=>v.play()));playing=true}catch(e){vs.forEach(v=>v.pause());playing=false}};
document.querySelector('#pause').onclick=()=>{playing=false;vs.forEach(v=>v.pause())};
document.querySelectorAll('[data-seek]').forEach(b=>b.onclick=()=>seek(Number(b.dataset.seek)));
function tick(){if(playing&&Math.abs(vs[1].currentTime-vs[0].currentTime)>.12)vs[1].currentTime=vs[0].currentTime;requestAnimationFrame(tick)}tick();
