#!/usr/bin/env python3
from __future__ import annotations
import json, os, re, subprocess, sys, tempfile
from pathlib import Path
import nbformat, yaml

ROOT = Path.cwd()
EXCLUDED = {'.git','.venv','venv','node_modules','build','dist','.pytest_cache','.ruff_cache','__pycache__'}
REQUIRED = {'README.md','LICENSE','LICENSE_REVIEW.md','THIRD_PARTY_NOTICES.md','PUBLIC_RELEASE_STATUS.md','OWNERSHIP_REVIEW.md'}
SECRETS = [
 re.compile(r'\bsk-(?:proj-)?[A-Za-z0-9_-]{16,}'),
 re.compile(r'\b(?:ghp_|gho_|ghu_|ghs_|ghr_|github_pat_)[A-Za-z0-9_]{16,}'),
 re.compile(r'\bAIza[0-9A-Za-z_-]{20,}'), re.compile(r'\bAKIA[0-9A-Z]{16}\b'),
 re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')]
errors=[]; warnings=[]

def run(cmd, timeout=300):
    p=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=timeout)
    if p.returncode:
        errors.append({'command':cmd,'returncode':p.returncode,'stdout':p.stdout[-12000:],'stderr':p.stderr[-12000:]})

def files():
    for p in ROOT.rglob('*'):
        if p.is_file() and not any(part in EXCLUDED for part in p.relative_to(ROOT).parts):
            yield p

for name in sorted(REQUIRED):
    if not (ROOT/name).is_file(): errors.append({'missing_required_file':name})

py=[]; sh=[]; c=[]; cpp=[]; java=[]; tests=False
for p in files():
    rel=str(p.relative_to(ROOT))
    size=p.stat().st_size
    if size>100*1024*1024: errors.append({'file_over_100_mb':rel,'bytes':size})
    if size<=5_000_000:
        text=p.read_text(encoding='utf-8',errors='ignore')
        for pattern in SECRETS:
            if pattern.search(text): errors.append({'credential_pattern':rel}); break
    if p.suffix=='.ipynb':
        try:
            nb=nbformat.read(p,as_version=4)
            for cell in nb.cells:
                if cell.cell_type=='code' and (cell.get('outputs') or cell.get('execution_count') is not None):
                    errors.append({'notebook_output_or_count':rel}); break
        except Exception as exc: errors.append({'malformed_notebook':rel,'error':str(exc)})
    elif p.suffix=='.json':
        try: json.loads(p.read_text(encoding='utf-8'))
        except Exception as exc: errors.append({'invalid_json':rel,'error':str(exc)})
    elif p.suffix in {'.yaml','.yml'}:
        try: yaml.safe_load(p.read_text(encoding='utf-8'))
        except Exception as exc: errors.append({'invalid_yaml':rel,'error':str(exc)})
    if p.suffix=='.py': py.append(str(p))
    if p.suffix in {'.sh','.bash'}: sh.append(str(p))
    if p.suffix=='.c': c.append(str(p))
    if p.suffix in {'.cpp','.cc','.cxx'}: cpp.append(str(p))
    if p.suffix=='.java': java.append(str(p))
    if re.search(r'(^|/)(test|tests)(/|$)|(^|/)test_.*\.py$',rel,re.I): tests=True

if py:
    run([sys.executable,'-m','compileall','-q','.'])
    run(['ruff','check','.','--exclude','notebooks,.git,.venv,venv,build,dist'])
    run(['black','--check','.','--exclude','/(notebooks|\.git|\.venv|venv|build|dist)/'])
    run(['bandit','-q','-r','.','-x','tests,notebooks,.venv,venv,build,dist'])
    if tests: run([sys.executable,'-m','pytest','-q'],timeout=600)
for p in sh: run(['bash','-n',p])
for p in c: run(['gcc','-std=c11','-fsyntax-only',p])
for p in cpp: run(['g++','-std=c++17','-fsyntax-only',p])
if (ROOT/'pom.xml').is_file(): run(['mvn','-q','test'],timeout=900)
elif (ROOT/'gradlew').is_file():
    os.chmod(ROOT/'gradlew',0o755); run(['./gradlew','test','--no-daemon'],timeout=900)
elif (ROOT/'build.gradle').is_file(): run(['gradle','test','--no-daemon'],timeout=900)
elif java:
    out=Path(tempfile.mkdtemp(prefix='javac-'))
    run(['javac','-d',str(out),*java],timeout=600)
manual=[]
own=ROOT/'OWNERSHIP_REVIEW.md'
if own.is_file() and '- [ ]' in own.read_text(encoding='utf-8',errors='ignore'):
    manual.append('OWNERSHIP_REVIEW.md contains unchecked human-only items')
report={'technical_errors':errors,'manual_warnings':manual,'technical_ready':not errors}
print(json.dumps(report,indent=2))
if errors: sys.exit(1)
