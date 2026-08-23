#!/usr/bin/env python3
"""Render the per-model x backend facts matrix (REPORT_sm90_v7.html).

v7 fixes over v6:
  - record selection prefers outcome.status == ok (a stale failed record for
    the same (repo, backend) no longer blanks the identity sub-lines)
  - kv line shows MEASURED resolution: sglang from resolved.kv_cache_dtype,
    vllm/trtllm from facts/kvcap/<id>.json sidecars (runner / KVCacheManager
    allocation dtype captured by the probes' kv_cache_resolved block)
"""
import json, os, pathlib, re, html, yaml
from collections import Counter

ROOT = pathlib.Path(os.environ.get("AIC_PROBE_WORKSPACE",
                                   pathlib.Path(__file__).resolve().parents[1]))

t = yaml.safe_load(open(ROOT / 'targets.yaml' if (ROOT / 'targets.yaml').exists()
                        else pathlib.Path(__file__).parent / 'targets.yaml'))
custom = {}
for fam in t['families'].values():
    for ck in fam.get('checkpoints') or []:
        for be, v in (ck.get('cli_extra_args') or {}).items():
            custom[(ck['repo'], be)] = ' '.join(v['args']) if isinstance(v, dict) else ' '.join(v)
    for repo, o in (fam.get('checkpoint_overrides') or {}).items():
        for be, v in (o.get('cli_extra_args') or {}).items():
            custom[(repo, be)] = ' '.join(v['args']) if isinstance(v, dict) else ' '.join(v)

recs = {}
for l in open(ROOT / 'archive/records.jsonl'):
    r = json.loads(l)
    k = (r['target'].get('repo'), r['runtime']['backend'])
    new_ok = (r.get('outcome') or {}).get('status') == 'ok'
    old_ok = k in recs and (recs[k].get('outcome') or {}).get('status') == 'ok'
    if new_ok or not old_ok:
        recs[k] = r

# (repo, backend) -> a kvcap sidecar for that identity (any probed sibling id)
KVCAP_ALT = {}
for _pf in ROOT.glob('archive/plan*.json'):
    for _run in json.loads(_pf.read_text()):
        if isinstance(_run, dict) and 'id' in _run:
            _p = ROOT / 'facts' / 'kvcap' / f"{_run['id']}.json"
            if _p.exists():
                KVCAP_ALT[(_run.get('repo'), _run.get('backend'))] = _p

SPECIAL = {
 ('moonshotai/Kimi-K3','sglang'): 'pin未注册·0.5.17已实测通',
 ('moonshotai/Kimi-K3','vllm'):   'pin未注册·0.27.1已实测通',
 ('moonshotai/Kimi-K3','trtllm'): 'rc23 quant解析墙·rc24已过(缺fla)',
 ('nvidia/Kimi-K3-NVFP4','sglang'): 'pin未注册·0.5.17支持',
 ('nvidia/Kimi-K3-NVFP4','vllm'):   'pin未注册·0.27.1支持',
 ('nvidia/Kimi-K3-NVFP4','trtllm'): 'FP8_PB_WO枚举缺失(rc24同)',
 ('MiniMaxAI/MiniMax-M3-MXFP8','sglang'): '能力声明锁死(marlin也绕不过)',
}

def failtag(repo, note):
    if re.search('not a valid Hugg', note): return 'generator拒绝'
    if re.search('Cannot find model module|not a registered|not supported for now|Unknown architecture|pydantic.*value_', note): return '框架未注册'
    if re.search('NotImplementedError', note): return 'tied-embed量化缺口'
    if re.search('Only gated SiLU', note): return 'NVFP4×gelu-MoE无路径'
    if re.search('pre-blackwell|Arch unsupported|use Blackwell|TllmGenFmhaRunner|Minimum ca|COMPRESS pool|No supported MoE GEMM tactic|mxfp8 is not supported|NVFP4 quantization with the selected', note): return '平台下限(需Blackwell)'
    if re.search('Mismatched Tensor', note): return 'flake(flashinfer,有env规避)'
    if re.search('sparse forward|KVCacheManagerV2', note): return 'rc23 M3-sparse未接线'
    if re.search('frame #|No valid attention backend', note): return 'ckpt强制fp8-KV'
    if re.search('NoneType|QuantAlgo', note): return 'quant解析器缺口'
    return 'framework缺口'

_SHORT = {'torch.bfloat16': 'bf16', 'torch.float16': 'fp16',
          'torch.float8_e4m3fn': 'fp8_e4m3', 'torch.uint8': 'fp8(u8)'}

def kv_measured(be, rid, rec):
    """configured -> actually-allocated kv dtype, measured only."""
    res = rec.get('resolved') or {}
    side = ROOT / 'facts' / 'kvcap' / f'{rid}.json'
    if not side.exists():  # sibling record of the same (repo, backend) identity
        alt = KVCAP_ALT.get((rec['target'].get('repo'), be))
        if alt: side = alt
    cfg = 'auto'
    if be == 'vllm':
        cfg = res.get('kv_cache_dtype') or 'auto'
    elif be == 'sglang':
        cli = (rec.get('runtime') or {}).get('engine_cli') or ''
        m = re.search(r'--kv-cache-dtype[= ]("?)([\w]+)\1', cli)
        cfg = m.group(2) if m else 'auto'
    act = '未捕获'
    if side.exists():
        kv = (json.loads(side.read_text()) or {}).get('kv_cache_resolved') or {}
        if be == 'vllm':
            act = kv.get('runner_kv_cache_dtype')
            cfg = kv.get('cache_config_dtype') or cfg
        elif be == 'sglang':
            act = kv.get('runner_kv_cache_dtype') or kv.get('pool.dtype')
            cfg = kv.get('server_arg') or cfg
        else:
            act = next((v for k, v in kv.items()
                        if k.startswith('manager.') and 'dtype' in k), None)
            cfg = (kv.get('configured') or cfg).replace('DataType.', '').lower()
        act = _SHORT.get(act, act) or '未捕获'
        act = str(act).replace('DataType.', '').replace('torch.', '').lower()
    elif be == 'sglang':  # pre-backfill fallback: server-level resolution
        act = res.get('kv_cache_dtype') or '?'
    return f"{cfg}→{act}"

def identity_lines(be, rid, r):
    ident = r.get('identity') or {}
    attn = (ident.get('attn_backend') or '').replace('Backend','').replace('_kernel','') or '—'
    moe_q = next((k for k in (ident.get('modules') or {}) if 'MoE' in k), None)
    moe_b = set()
    for op in r.get('ops') or []:
        if re.search(r'moe|Marlin|marlin|grouped|expert', ' '.join(op.get('kernels') or [])):
            moe_b |= set(op.get('backends') or [])
    moe_b -= {'cublas','vllm_kernel','sgl_kernel','torch','infra','framework_native'}
    moe_txt = '—'
    if moe_q:
        q = moe_q.replace('Method','').replace('FusedMoE','').replace('MoE','')
        warn = ' ⚠静默降级' if 'Unquantized' in moe_q and 'FP4' in r['target'].get('repo','').upper() else ''
        moe_txt = (q or 'MoE') + ('→' + '/'.join(sorted(moe_b)) if moe_b else '') + warn
    return attn[:26], moe_txt[:38], kv_measured(be, rid, r)[:26]

plans = {'sglang':'plan_roster_sgl.json','vllm':'plan_roster_vllm.json','trtllm':'plan_roster_trt.json'}
out, order, cnt = {}, [], Counter()
for be, pf in plans.items():
    for r in json.load(open(ROOT / f'archive/{pf}')):
        repo = r.get('repo')
        if repo and repo not in order: order.append(repo)
        if 'skip' in r:
            cell = {'st':'fail','tag':'generator拒绝','note':r['skip'][:110]}
        else:
            raw = ROOT / f"archive/raw/{r['id']}.json"
            if not raw.exists():
                cell = {'st':'fail','tag':'容量(需tp8)','note':'capacity known_bad'}
            else:
                f = json.loads(raw.read_text()); e = f.get('errors') or {}
                rec = recs.get((repo, be))
                if not e:
                    st = 'passc' if (repo, be) in custom else 'pass'
                    if rec and (rec.get('outcome') or {}).get('status') == 'ok':
                        a, mo, kv = identity_lines(be, rec['id'], rec)
                    else:
                        a, mo, kv = '—','—','—'
                    cell = {'st':st,'attn':a,'moe':mo,'kv':kv,'cust':custom.get((repo,be))}
                else:
                    note = next(iter(e.values())).strip().splitlines()[-1][:120]
                    tag = SPECIAL.get((repo, be)) or failtag(repo, note)
                    cell = {'st':'fail','tag':tag,'note':note}
        out.setdefault(repo, {})[be] = cell
        cnt[(be, cell['st'])] += 1

BE = ['sglang','vllm','trtllm']
BEHDR = {'sglang':'sglang 0.5.16','vllm':'vLLM 0.24.0','trtllm':'TRT-LLM 1.3.0rc23'}
rows = []
for repo in order:
    tds = ''
    for be in BE:
        c = out.get(repo,{}).get(be)
        if not c: tds += '<td class="c na">—</td>'; continue
        if c['st'] in ('pass','passc'):
            cls = 'pc' if c['st']=='passc' else 'ok'
            head = '✓ pass' + ('+定制' if c['st']=='passc' else '')
            tds += (f"<td class='c {cls}' title='{html.escape(c.get('cust') or '')}'>{head}"
                    f"<span class='sub'>attn {html.escape(c['attn'])}</span>"
                    f"<span class='sub'>moe {html.escape(c['moe'])}</span>"
                    f"<span class='sub'>kv {html.escape(c['kv'])}</span></td>")
        else:
            tds += f"<td class='c fact' title='{html.escape(c['note'])}'>✗ <span class='tag'>{c['tag']}</span></td>"
    rows.append(f"<tr><td class='m'>{html.escape(repo)}</td>{tds}</tr>")

S = {be: {k: cnt.get((be,k),0) for k in ('pass','passc','fail')} for be in BE}
N = len(order)
body = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>op-probe facts v7 — kv 实测决议 (2026-08-23)</title>
<style>
:root{{--surface:#fcfcfb;--ink:#1f1f1e;--ink2:#5f5e5a;--muted:#8a8984;--line:#e7e6e2;
--good:#0ca30c;--goodbg:#eef8ee;--cust:#1a7f9c;--custbg:#e8f4f8;--crit:#d03b3b;--critbg:#fbeaea;--chip:#f4f3f0}}
@media (prefers-color-scheme: dark){{:root{{--surface:#1a1a19;--ink:#f0efec;--ink2:#b5b4ae;
--line:#33322f;--goodbg:#12300f;--custbg:#0e2a33;--critbg:#3a1414;--chip:#262624}}}}
*{{box-sizing:border-box}} body{{margin:auto;background:var(--surface);color:var(--ink);max-width:1280px;
font:14px/1.45 -apple-system,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif;padding:32px 40px 64px}}
h1{{font-size:20px;margin:0 0 4px}} .meta{{color:var(--ink2);margin:0 0 22px;font-size:13px}}
h2{{font-size:15px;margin:28px 0 10px}}
table{{border-collapse:collapse;width:100%;font-size:12.5px}}
th{{text-align:left;color:var(--ink2);font-weight:600;border-bottom:2px solid var(--line);padding:6px 8px;position:sticky;top:0;background:var(--surface)}}
td{{border-bottom:1px solid var(--line);padding:4px 8px;vertical-align:top}}
td.m{{font-family:ui-monospace,Menlo,monospace;font-size:12px;white-space:nowrap}}
td.c{{white-space:nowrap;line-height:1.35}} td.ok{{color:var(--good);background:var(--goodbg)}}
td.pc{{color:var(--cust);background:var(--custbg)}}
td.fact{{color:var(--crit);background:var(--critbg)}} td.na{{color:var(--muted)}}
td .tag{{font-size:11.5px}} td .sub{{display:block;color:var(--ink2);font-size:10.5px;font-family:ui-monospace,Menlo,monospace}}
tr:hover td{{filter:brightness(.97)}} .sum td{{font-weight:650;font-size:14px}}
code{{background:var(--chip);border-radius:4px;padding:1px 5px;font-size:12px}}
.note{{color:var(--ink2);font-size:12.5px}}
</style></head><body>
<h1>op-probe facts v7 — kv 实测决议</h1>
<p class="meta">2026-08-23 · {N} 模型 × 3 后端 · SM90 (H20) · kv 行为实测: 配置值→框架实际分配的 dtype · PR #1572</p>
<h2>结论总表</h2>
<table style="max-width:600px"><thead><tr><th>后端</th><th>pass</th><th>pass+定制</th><th>fail</th></tr></thead>
<tbody class="sum">
<tr><td>{BEHDR['sglang']}</td><td class="ok">{S['sglang']['pass']}</td><td class="pc">{S['sglang']['passc']}</td><td class="fact">{S['sglang']['fail']}</td></tr>
<tr><td>{BEHDR['vllm']}</td><td class="ok">{S['vllm']['pass']}</td><td class="pc">{S['vllm']['passc']}</td><td class="fact">{S['vllm']['fail']}</td></tr>
<tr><td>{BEHDR['trtllm']}</td><td class="ok">{S['trtllm']['pass']}</td><td class="pc">{S['trtllm']['passc']}</td><td class="fact">{S['trtllm']['fail']}</td></tr>
</tbody></table>
<h2>读法</h2>
<ul class="note" style="padding-left:20px">
<li><code>kv a→b</code>: a=命令/引擎配置值, b=<b>实测</b>框架实际分配的 KV dtype(vllm 取 model runner 的 kv_cache_spec, trtllm 取 KVCacheManager, sglang 取 server resolved)。同一 checkpoint 同一 auto 可能落不同精度——如 MiniMax-M2: sglang auto→fp8_e4m3(继承 ckpt hf_quant), vllm auto→bf16(跟模型 dtype)。</li>
<li><code>moe</code> 箭头两侧不一致 = <b>身份≠计算精度</b>: 如 <code>ModelOptNvFp4→marlin</code> 是 NVFP4 身份、Marlin w4a16 反量化计算(SM90 无 fp4 tensor core)。</li>
<li>量化 checkpoint 上出现 <code>Unquantized ⚠静默降级</code> = 框架静默丢弃量化(M3-NVFP4×sglang)。</li>
<li>ckpt 元数据强制的 fp8-KV 绕过配置链(红格 'ckpt强制fp8-KV')。fail 红格悬停看原始报错; 定制格悬停看参数。</li>
</ul>
<h2>逐模型矩阵</h2>
<table><thead><tr><th>checkpoint</th><th>{BEHDR['sglang']}</th><th>{BEHDR['vllm']}</th><th>{BEHDR['trtllm']}</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<p class="note" style="margin-top:22px">facts: archive/records.jsonl + raw/ + facts/kvcap/ · 命令: golden/&lt;id&gt;/command.txt · PR #1572</p>
</body></html>"""
outp = ROOT / 'facts/REPORT_sm90_v7.html'
outp.write_text(body)
print('wrote', outp, len(body)//1024, 'KB', len(rows), 'rows', S)
