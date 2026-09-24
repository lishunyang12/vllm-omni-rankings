/* Title/label rules shared by the browser and Node verification. */
(function (root) {
  'use strict';
  const groups = [
    {id:'model',name:'Models',hint:'Models mentioned in titles or labels'},
    {id:'hardware',name:'Hardware',hint:'Mentioned platforms, not verified compatibility'},
    {id:'area',name:'Areas',hint:'PRs can span multiple areas'}
  ];
  const make=(id,group,name,regex,labels=[])=>({id,group,name,regex,labels});
  const topics=[
    make('qwen-omni','model','Qwen Omni',/\bqwen[\d.\s-]*omni\b/i),
    make('qwen-image','model','Qwen Image',/\bqwen[\d.\s-]*image\b/i),
    make('qwen-tts','model','Qwen TTS',/\bqwen[\d.\s-]*tts\b/i),
    make('wan','model','Wan',/\bwan(?=\d|\b)/i),
    make('minimax-h3','model','MiniMax-H3',/\bminimax[\s-]*h3\b|\bh3\b/i),
    make('hunyuan','model','Hunyuan',/\bhunyuan/i),
    make('flux','model','FLUX',/\bflux\b/i),
    make('minicpm','model','MiniCPM-o',/\bminicpm/i),
    make('glm','model','GLM',/\bglm(?:\b|\d)/i),
    make('cosmos','model','Cosmos',/\bcosmos(?:\b|\d)/i),
    make('bagel','model','BAGEL',/\bbagel\b/i),
    make('ltx','model','LTX',/\bltx(?:\b|\d)/i),
    make('moss','model','MOSS',/\bmoss(?:\b|\d)/i),
    make('cosyvoice','model','CosyVoice',/\bcosy[\s-]*voice/i),
    make('sensenova','model','SenseNova',/\bsensenova/i),
    make('sana','model','SANA',/\bsana\b/i),
    make('seedvr','model','SeedVR',/\bseedvr/i),
    make('step','model','Step / NextStep',/\bnextstep|\bstep[\s-]*(?:audio|video|fun|\d)/i),
    make('stable-diffusion','model','Stable Diffusion',/\bstable[\s-]*diffusion\b|\bsdxl\b|\bsd3(?:\b|\.)/i),
    make('z-image','model','Z-Image',/\bz[\s-]*image\b/i),
    make('longcat','model','LongCat',/\blongcat/i),
    make('omnivoice','model','OmniVoice',/\bomnivoice/i),
    make('higgs','model','Higgs Audio',/\bhiggs\b/i),
    make('ovis','model','Ovis',/\bovis\b/i),
    make('joyai','model','JoyAI',/\bjoyai/i),
    make('mammoth','model','MammothModa',/\bmammoth[\s-]*moda/i),
    make('personaplex','model','PersonaPlex',/\bpersonaplex/i),
    make('ernie','model','ERNIE',/\bernie\b/i),
    make('lingbot','model','LingBot',/\blingbot/i),
    make('magi','model','MAGI',/\bmagi(?:\b|\d)/i),
    make('lance','model','Lance',/\blance\b/i),
    make('helios','model','Helios',/\bhelios\b/i),
    make('dreamzero','model','DreamZero',/\bdreamzero\b/i),
    make('anima','model','Anima',/\banima\b/i),
    make('yue','model','YuE',/\byue(?:\b|\d)/i),
    make('omnigen','model','OmniGen',/\bomnigen/i),
    make('nemotron','model','Nemotron',/\bnemotron\b/i),
    make('ming','model','Ming',/\bming[\s-]*(?:image|omni|flash)\b/i),
    make('cuda','hardware','NVIDIA / CUDA',/\bcuda\b|\bnvidia\b|\btensorrt\b|\b(?:a100|h100|h200|b200|b300|blackwell|hopper)\b/i,['cuda-test']),
    make('rocm','hardware','AMD / ROCm',/\b(?:rocm|amd|mi300\w*|mi350\w*|hip)\b/i,['ROCm','amd-test']),
    make('ascend','hardware','Ascend / NPU',/\b(?:ascend|npu|npugraph|cann|mindie)\b/i,['NPU','npu-test']),
    make('xpu','hardware','Intel / XPU',/\b(?:intel|xpu|oneapi|sycl)\b/i,['xpu','intel-test']),
    make('cpu','hardware','CPU',/\bcpu\b/i),
    make('apple','hardware','Apple / MPS',/\b(?:apple|mps|metal)\b/i),
    make('attention','area','Attention',/attention|\bsdpa\b|\bqkv\b|\bqk\b|\brope\b/i),
    make('quantization','area','Quantization / low precision',/quantiz|\bquant\b|\b(?:fp8|fp4|nvfp4|mxfp8|int8|int4|w4|w8a8|awq|gptq)\b|bitsandbytes|autoround|svdquant/i,['quantization']),
    make('cache','area','Caching / KV cache',/cache|\bcaching\b|\bringkv\b/i),
    make('parallel','area','Distributed / parallel',/parallel|distributed|multi[\s-]*node|allgather|allreduce|ulysses|\b(?:usp|hsdp|fsdp|tp|sp|dp|dcp|cfg|rdma|nccl|nixl|mooncake|ray)\b/i),
    make('performance','area','Inference performance',/\[(?:perf|performance|optimization)\]|speed[\s-]*up|\bfaster\b|\boptimi[sz]\w*|\blatency\b|\bthroughput\b|torch\.compile|cuda[\s-]*graph/i,['Kernel optimization']),
    make('kernels','area','Kernels / operators',/\bkernel\w*|\bfus(?:e|ed|ion)\b|\btriton\b|\bcutlass\b|\bflashinfer\b|\bswiglu\b|\brmsnorm\b/i,['Kernel optimization']),
    make('memory','area','Memory / offload',/\bmemory\b|\bvram\b|\boom\b|offload|zero[\s-]*copy|\btiling\b|\btiled\b|residency/i),
    make('streaming','area','Streaming / full duplex',/stream(?:ing)?\b|duplex|realtime|real[\s-]*time|websocket|\bplayback\b|\bbarge[\s-]*in\b/i),
    make('audio','area','Speech / audio',/tts\b|\basr\b|\bstt\b|audio|speech|voice|talker|token2wav|\bcodec\b|\bmusic\b/i,['tts']),
    make('image','area','Image generation',/image|text[\s-]*to[\s-]*image|\bt2i\b|\bi2i\b|inpaint|outpaint/i),
    make('video','area','Video generation',/video|\bt2v\b|\bi2v\b|\bv2v\b|\bti2v\b/i),
    make('diffusion','area','Diffusion / VAE',/diffusion|diffusers|\bvae\b|denois|\bdit\b|\bflow[\s-]*match/i,['diffusion']),
    make('serving','area','Serving / API',/serving|\bserver\b|\bapi\b|endpoint|openai[\s-]*compat|frontend|comfyui|gradio|\bhttp\b|\bclient\b/i,['frontend']),
    make('engine','area','Engine / scheduling',/\[core\]|\bengine\b|schedul|pipeline|\bbatch(?:ing)?\b|\bstage[\s-]*\d|multi[\s-]*stage|worker|orchestrat/i,['core']),
    make('lora','area','LoRA',/\blora\b/i),
    make('weights','area','Model loading / weights',/\bload(?:er|ing)?\b|\bweight\w*|\bcheckpoint\w*|\bsafetensors\b/i),
    make('observability','area','Benchmarks / observability',/benchmark|profil|metrics|telemetry|\blogg(?:ing|er)\b/i,['benchmark/profiler/metrics/logger']),
    make('docs','area','Docs / examples',/\[(?:doc|docs|example|examples|recipe)\]|\bdocs?\b|documentation|\breadme\b|\btutorial\b|\bexample\w*/i,['documentation']),
    make('testing','area','Tests / build',/\[(?:ci|ci\/build|ci\/test|test|tests)\]|\bci\b|\btests?\b|\btesting\b|\bpytest\b|\bnightly\b|\bbuild\b|\bpackag(?:e|ing)\b/i,['ci','ci/build','CI/CD']),
    make('refactor','area','Architecture / refactoring',/\brefactor\w*|\bmigrat\w*|\bdeprecat\w*|\bconsolidat\w*/i,['refactor']),
    make('model-support','area','Model integration',/\[(?:model|new model)\]/i,['new model']),
    make('training','area','Training / RL',/\b(?:rl|grpo|flowgrpo|training|finetun\w*)\b/i,['RL']),
    make('robotics','area','World models / VLA',/\bworld\b|\bvla\b|\brobot\w*/i,['world model','VLA'])
  ];
  const normalize=text=>String(text||'').replace(/[_‐‑–—]/g,'-');
  function classify(pr) {
    const title=normalize(pr.title), labels=(pr.labels||[]).map(String);
    return topics.flatMap(topic=>{
      const hit=topic.regex.exec(title);
      // Status labels like ci-failure must not imply a CI infrastructure change.
      const labelHits=labels.filter(label=>topic.labels.some(value=>value.toLowerCase()===label.toLowerCase()) ||
        (topic.group!=='area' && topic.regex.test(normalize(label))));
      if(!hit && !labelHits.length)return [];
      return [{id:topic.id,evidence:[
        ...(hit?['Title: '+hit[0]]:[]),...labelHits.map(label=>'Label: '+label)
      ]}];
    });
  }
  const api={groups,topics,classify,version:1};
  if(typeof module!=='undefined' && module.exports)module.exports=api;
  else root.OmniTopics=api;
})(typeof globalThis!=='undefined'?globalThis:this);
