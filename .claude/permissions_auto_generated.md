# Auto-Generated Permissions Update

**Generated**: 2026-01-01 12:05:34
**Rules Added**: 52

## What Happened

The Permission Auto-Learner analyzed your recent approval patterns and detected
high-confidence permission patterns that will reduce future prompts.

## Added Permissions

### Allow POSIX search/transform commands

**Confidence**: 99.8%  
**Occurrences**: 125  
**Impact**: Low impact: ~1% fewer prompts  

**Permissions added**:
- `Bash(grep:*)`
- `Bash(awk:*)`
- `Bash(sed:*)`

### Allow POSIX filesystem commands (find, ls, mkdir, etc.)

**Confidence**: 97.5%  
**Occurrences**: 297  
**Impact**: Low impact: ~3% fewer prompts  

**Permissions added**:
- `Bash(find:*)`
- `Bash(ls:*)`
- `Bash(mkdir:*)`

### Allow POSIX file reading/stats commands

**Confidence**: 97.3%  
**Occurrences**: 35  
**Impact**: Low impact: ~0% fewer prompts  

**Permissions added**:
- `Bash(cat:*)`
- `Bash(head:*)`
- `Bash(tail:*)`

### Allow Standard git workflow commands (excludes force/hard operations)

**Confidence**: 96.1%  
**Occurrences**: 109  
**Impact**: Low impact: ~1% fewer prompts  

**Permissions added**:
- `Bash(git:*)`

### Allow Full project directory access

**Confidence**: 96.0%  
**Occurrences**: 311  
**Impact**: Low impact: ~3% fewer prompts  

**Permissions added**:
- `Read(/home/*/project/**)`
- `Write(/home/*/project/**)`

### Allow File read operations

**Confidence**: 95.2%  
**Occurrences**: 607  
**Impact**: Low impact: ~7% fewer prompts  

**Permissions added**:
- `Read(/**)`
- `Glob(**)`

### Allow Cloud provider CLIs (GCP, AWS, Azure)

**Confidence**: 95.0%  
**Occurrences**: 31  
**Impact**: Low impact: ~0% fewer prompts  

**Permissions added**:
- `Bash(gcloud:*)`
- `Bash(aws:*)`
- `Bash(az:*)`

### Allow Read-only git commands

**Confidence**: 94.1%  
**Occurrences**: 19  
**Impact**: Low impact: ~0% fewer prompts  

**Permissions added**:
- `Bash(git status:*)`
- `Bash(git log:*)`
- `Bash(git diff:*)`
- `Bash(git show:*)`
- `Bash(git branch:*)`

### Allow Nix/NixOS ecosystem tools

**Confidence**: 92.9%  
**Occurrences**: 29  
**Impact**: Low impact: ~0% fewer prompts  

**Permissions added**:
- `Bash(nix:*)`
- `Bash(devenv:*)`

### Allow Language runtime interpreters

**Confidence**: 90.9%  
**Occurrences**: 19  
**Impact**: Low impact: ~0% fewer prompts  

**Permissions added**:
- `Bash(python:*)`
- `Bash(node:*)`

### Allow Network/HTTP client tools

**Confidence**: 90.7%  
**Occurrences**: 23  
**Impact**: Low impact: ~0% fewer prompts  

**Permissions added**:
- `Bash(curl:*)`
- `Bash(xh:*)`

### Allow File write/edit operations

**Confidence**: 90.3%  
**Occurrences**: 484  
**Impact**: Low impact: ~5% fewer prompts  

**Permissions added**:
- `Write(/**)`
- `Edit(/**)`

### Allow Shell built-ins and utilities

**Confidence**: 89.5%  
**Occurrences**: 22  
**Impact**: Low impact: ~0% fewer prompts  

**Permissions added**:
- `Bash(echo:*)`
- `Bash(which:*)`

### Allow Modern CLI tools (fd, eza, bat, rg, etc.)

**Confidence**: 88.8%  
**Occurrences**: 31  
**Impact**: Low impact: ~0% fewer prompts  

**Permissions added**:
- `Bash(fd:*)`
- `Bash(eza:*)`
- `Bash(bat:*)`
- `Bash(rg:*)`
- `Bash(dust:*)`
- `Bash(procs:*)`

### Allow Database CLI tools

**Confidence**: 88.7%  
**Occurrences**: 18  
**Impact**: Low impact: ~0% fewer prompts  

**Permissions added**:
- `Bash(sqlite3:*)`
- `Bash(psql:*)`
- `Bash(mycli:*)`

### Allow GitHub CLI commands (gh pr, gh issue, gh api, etc.)

**Confidence**: 87.0%  
**Occurrences**: 8  
**Impact**: Low impact: ~0% fewer prompts  

**Permissions added**:
- `Bash(gh:*)`

### Allow Package manager commands

**Confidence**: 80.2%  
**Occurrences**: 9  
**Impact**: Low impact: ~0% fewer prompts  

**Permissions added**:
- `Bash(npm:*)`
- `Bash(pnpm:*)`
- `Bash(pip:*)`
- `Bash(uv:*)`

### Allow Test execution commands

**Confidence**: 70.2%  
**Occurrences**: 8  
**Impact**: Low impact: ~0% fewer prompts  

**Permissions added**:
- `Bash(npm test:*)`
- `Bash(pytest:*)`
- `Bash(cargo test:*)`
- `Bash(go test:*)`

### Allow Pytest test execution

**Confidence**: 60.5%  
**Occurrences**: 6  
**Impact**: Low impact: ~0% fewer prompts  

**Permissions added**:
- `Bash(pytest:*)`
- `Bash(python -m pytest:*)`

### Allow Ruff linter/formatter

**Confidence**: 60.5%  
**Occurrences**: 6  
**Impact**: Low impact: ~0% fewer prompts  

**Permissions added**:
- `Bash(ruff:*)`

---

These permissions have been automatically added to `.claude/settings.local.json`.
You can review and modify them at any time.

To disable auto-generation, set:
```json
{
  "_auto_generated_permissions": {
    "enabled": false
  }
}
```
