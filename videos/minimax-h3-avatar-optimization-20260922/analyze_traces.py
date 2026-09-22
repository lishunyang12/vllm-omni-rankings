"""Read preserved Nsight SQLite exports without contacting a GPU or service.

CUDA/NVTX totals describe instrumented traces. They are never E2E latency.
"""
import argparse, collections, csv, hashlib, json, sqlite3, statistics
from pathlib import Path
def trace_catalog(workspace,temporary_root):
 return {
  'config_a_15s':workspace/'minimax-h3-native/experiments/fasth3-vae-lossless-overlap-20260918/traces/fasth3-config-a-vae-lossless-hardware-metrics.sqlite',
  'bf16_prefetch_1088':temporary_root/'h3-infra-1088-20260921/request-cuda.sqlite',
  'bf16_resident_online_mx_1088':temporary_root/'h3-infra-1088-20260921/resident-cuda.sqlite',
  'all_bf16_weights_1088':temporary_root/'h3-bf16-comparison-20260921-01/weights-cuda.sqlite',
  **{f'cache_{n}':temporary_root/f'h3-lotus-20260921/cache-capture.{n}.sqlite' for n in range(1,5)},
 }
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def union_ns(intervals):
 total=0;left=right=None
 for a,b in sorted(intervals):
  if right is None:left,right=a,b
  elif a>right:total+=right-left;left,right=a,b
  else:right=max(right,b)
 return total+(right-left if right is not None else 0)
def analyze(path):
 c=sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
 names=dict(c.execute('SELECT id,value FROM StringIds'))
 groups=collections.defaultdict(list)
 for r in c.execute('SELECT start,end,text,textId,globalTid FROM NVTX_EVENTS WHERE end IS NOT NULL'):
  name=r['text'] or names.get(r['textId'],'')
  if name.startswith('h3.profile.'):
   groups[name].append((r['end']-r['start'])/1e6)
 stages={k:dict(count=len(v),median_ms=statistics.median(v),min_ms=min(v),max_ms=max(v),sum_across_ranges_ms=sum(v)) for k,v in groups.items()}
 kernels=collections.defaultdict(list);top=collections.defaultdict(lambda:[0,0])
 for r in c.execute('SELECT start,end,deviceId,demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL'):
  kernels[r['deviceId']].append((r['start'],r['end']))
  name=names[r['demangledName']];top[name][0]+=1;top[name][1]+=r['end']-r['start']
 device=[]
 for k,v in sorted(kernels.items()):
  span=max(b for a,b in v)-min(a for a,b in v)
  device.append(dict(device_id=k,kernel_calls=len(v),kernel_sum_ms=sum(b-a for a,b in v)/1e6,kernel_interval_union_ms=union_ns(v)/1e6,first_to_last_kernel_span_ms=span/1e6))
 kinds={r['id']:r['label'] for r in c.execute('SELECT * FROM ENUM_CUDA_MEMCPY_OPER')}
 mem=collections.defaultdict(lambda:[0,0,0,0,0,0])
 for r in c.execute('SELECT start,end,deviceId,bytes,copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY'):
  g=mem[(r['deviceId'],kinds[r['copyKind']])];g[0]+=1;g[1]+=r['bytes'];g[2]+=r['end']-r['start']
  if r['bytes']>10_000_000:g[3]+=1;g[4]+=r['bytes'];g[5]+=r['end']-r['start']
 transfers=[dict(device_id=k[0],kind=k[1],calls=v[0],bytes=v[1],gpu_sum_ms=v[2]/1e6,over_10MB_calls=v[3],over_10MB_bytes=v[4],over_10MB_gpu_sum_ms=v[5]/1e6) for k,v in sorted(mem.items())]
 result=dict(sqlite=str(path),sqlite_bytes=path.stat().st_size,sqlite_sha256=sha(path),nvtx_cpu_ranges=stages,devices=device,transfers=transfers,top_kernels=[dict(name=k,calls=v[0],gpu_sum_ms=v[1]/1e6) for k,v in sorted(top.items(),key=lambda x:x[1][1],reverse=True)[:20]],gpu_inventory=[dict(r) for r in c.execute('SELECT id,name,chipName,totalMemory,smCount,computeMajor,computeMinor FROM TARGET_INFO_GPU')])
 c.close();return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path(__file__).parent/'evidence/nsys-reanalysis.json');p.add_argument('--workspace',type=Path,required=True);p.add_argument('--temporary-root',type=Path,default=Path('/tmp'));args=p.parse_args()
 out={'method':'Read-only reanalysis of historical instrumented traces; no new GPU capture. NVTX ranges include CPU submission/waits; nested ranges and devices cannot be summed into E2E savings. Kernel interval union includes communication wait and is not SM utilization.', 'traces':{}}
 for label,path in trace_catalog(args.workspace,args.temporary_root).items():
  result=analyze(path)
  result['sqlite']=str(path).replace(str(args.workspace),'${WORKSPACE}').replace(str(args.temporary_root),'${TMP}')
  out['traces'][label]=result;print(label,'analyzed',flush=True)
 args.output.write_text(json.dumps(out,indent=2)+'\n')
 with args.output.with_suffix('.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['trace','range','count','median_ms','min_ms','max_ms','sum_across_ranges_ms'])
  for label,d in out['traces'].items():
   for name,v in d['nvtx_cpu_ranges'].items():w.writerow([label,name,*[v[k] for k in ['count','median_ms','min_ms','max_ms','sum_across_ranges_ms']]])
if __name__=='__main__':main()
