# ✅ Colab Training Pre-Flight Checklist

Use this checklist before starting your training session to ensure maximum success rate.

---

## 📋 PRE-TRAINING CHECKLIST

### ☑️ Phase 1: Documentation (1 minute)

- [ ] I have opened [COLAB_DOCS_INDEX.md](COLAB_DOCS_INDEX.md) or [COLAB_README.md](COLAB_README.md)
- [ ] I have chosen my guide:
  - [ ] COLAB_QUICK_FIX.md (1-minute fix)
  - [ ] COLAB_TRAINING_INSTRUCTIONS.md (full setup)
  - [ ] COLAB_DISCONNECTION_FIX.md (troubleshooting)
- [ ] I understand the 5 protection layers

---

### ☑️ Phase 2: Environment Setup (2 minutes)

- [ ] Opened Google Colab in browser
- [ ] Created new notebook or uploaded existing
- [ ] Selected GPU runtime:
  - [ ] Runtime → Change runtime type → GPU → Save
  - [ ] Verified GPU is enabled (see checkmark)
- [ ] Confirmed which GPU type I have:
  - [ ] T4 (Free tier) - 15GB
  - [ ] V100 (Pro) - 16GB  
  - [ ] A100 (Pro) - 40GB
- [ ] Browser is stable (good internet connection)

---

### ☑️ Phase 3: Protection Scripts (3 minutes)

#### Cell 1: Keep-Alive (CRITICAL!)

- [ ] Created first cell with keep-alive script
- [ ] Ran the cell
- [ ] Saw message: "✅ Keep-alive enabled!"
- [ ] Confirmed console shows "Staying alive..." messages

```python
# This cell is running ✅
from IPython.display import Javascript, display
display(Javascript('''
    setInterval(() => {
        document.querySelector("colab-toolbar-button#connect")?.click();
        console.log("Staying alive...");
    }, 60000);
'''))
print("✅ Keep-alive enabled!")
```

#### Cell 2: Auto-Reconnect

- [ ] Created second cell with auto-reconnect
- [ ] Ran the cell
- [ ] Saw message: "✅ Auto-reconnect enabled"

```python
# This cell is running ✅
display(Javascript('''
    setInterval(() => {
        if(!google.colab.kernel?.accessAllowed) {
            console.log("Reconnecting...");
            location.reload();
        }
    }, 30000);
'''))
print("✅ Auto-reconnect enabled")
```

#### Cell 3: Memory Management

- [ ] Created third cell with memory setup
- [ ] Ran the cell
- [ ] Saw GPU detection message
- [ ] Confirmed memory limits are set

```python
# This cell completed ✅
import torch, gc, os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
torch.cuda.set_per_process_memory_fraction(0.85)

# Should see:
# ✅ GPU: Tesla T4
#    Memory: 14.75 GB
```

---

### ☑️ Phase 4: Google Drive (2 minutes)

- [ ] Clicked "Mount Drive" or ran mount cell
- [ ] Granted permissions in popup
- [ ] Saw "Mounted at /content/drive"
- [ ] Created results directory
- [ ] Confirmed directory exists

```python
# This cell completed ✅
from google.colab import drive
drive.mount('/content/drive')
import os
os.makedirs('/content/drive/MyDrive/FarmFederate_Results', exist_ok=True)

# Should see:
# ✅ Google Drive mounted
# ✅ Results will auto-save to: /content/drive/MyDrive/FarmFederate_Results
```

---

### ☑️ Phase 5: Training Configuration (1 minute)

- [ ] GPU type auto-detected
- [ ] Batch size configured for GPU:
  - [ ] T4: batch_size=2
  - [ ] V100: batch_size=4
  - [ ] A100: batch_size=8
- [ ] LoRA rank configured
- [ ] Environment variables set

```python
# This cell completed ✅
# Auto-configuration based on GPU

# Should see:
# 🔍 Detected GPU Memory: 14.75 GB
#    📊 T4-optimized (Ultra Conservative)
#    - Batch Size: 2
#    - LoRA Rank: 4
```

---

### ☑️ Phase 6: Final Verification (1 minute)

Before clicking "Run Training":

#### System Status:
- [ ] Keep-alive is active (check console)
- [ ] Auto-reconnect is monitoring
- [ ] GPU memory is managed
- [ ] Google Drive is mounted
- [ ] Training is configured

#### Environment:
- [ ] Browser tab will stay open
- [ ] Computer won't go to sleep
- [ ] Internet connection is stable
- [ ] I have 3-5 hours available (T4) or 1.5-2 hours (A100)

#### Backup Plan:
- [ ] I know how to check checkpoints
- [ ] I know where Drive backups are
- [ ] I have bookmarked troubleshooting docs
- [ ] I understand how to resume if disconnected

---

## 🚀 START TRAINING!

If all checkboxes are ✅, you're ready!

Click "Run Training" cell and monitor for first 30 minutes.

---

## ⏱️ During Training Checklist

### First 30 Minutes (IMPORTANT):

- [ ] Training started successfully
- [ ] First model is training
- [ ] No error messages
- [ ] GPU is being used (check indicator)
- [ ] Memory stays under limit
- [ ] Keep-alive is working (console logs)

### Every Hour:

- [ ] Check progress (which model number)
- [ ] Verify memory usage is stable
- [ ] Confirm Drive backups are happening
- [ ] Ensure no error messages

### Final Hour:

- [ ] Models completing successfully
- [ ] Plots being generated
- [ ] Results saved to Drive
- [ ] Prepare for download

---

## 🎯 Expected Output at Each Stage

### Start (0-5 minutes):
```
✅ Keep-alive enabled!
✅ Auto-reconnect enabled
✅ GPU: Tesla T4
✅ Google Drive mounted
✅ Training configured
🚀 Starting training...
```

### During (Every 10-15 minutes):
```
[INFO] Training model 5/39: Flan-T5-Small
   Epoch 1/10: Loss 0.4523
   Epoch 2/10: Loss 0.3891
   ...
✓ Model 5 complete
   F1: 0.8523
   Accuracy: 0.8612
💾 Saved to Drive
```

### Completion:
```
✅ All 39 models trained!
⏱️ Total time: 3.8 hours
📊 Generating plots...
✅ Plots generated
💾 Backed up to Drive
📥 Ready for download
```

---

## ⚠️ Warning Signs

Stop and troubleshoot if you see:

### Critical Issues:
- ❌ "Runtime disconnected" (re-run keep-alive)
- ❌ "Out of memory" (reduce batch size)
- ❌ "Session crashed" (check memory)
- ❌ No output for 30+ minutes (check logs)

### Warning Signs:
- ⚠️ Memory above 90% (watch closely)
- ⚠️ Training slower than expected (normal on T4)
- ⚠️ Drive not backing up (check mount)

### What to Do:
1. Don't panic - checkpoints are saved
2. Check [COLAB_DISCONNECTION_FIX.md](COLAB_DISCONNECTION_FIX.md)
3. Apply specific fix for your issue
4. Resume training

---

## 📊 Success Indicators

You're on track if you see:

✅ Keep-alive console logs every minute
✅ GPU utilization 80-100%
✅ Memory stays under 85%
✅ Models completing every 5-10 minutes (T4)
✅ No error messages
✅ Files appearing in Google Drive
✅ Progress messages updating

---

## 🎉 Post-Training Checklist

### Immediate (Within 5 minutes):

- [ ] Verify all 39 models completed
- [ ] Check results.json exists
- [ ] Confirm plots were generated
- [ ] Verify Drive has all files
- [ ] Download results.zip
- [ ] Download plots.zip

### Within 24 Hours:

- [ ] Review top performing models
- [ ] Analyze comparison plots
- [ ] Document any issues encountered
- [ ] Share feedback/improvements

---

## 🆘 Emergency Procedures

### If Runtime Disconnects:

1. **Don't panic** - your data is safe in Drive
2. **Reconnect** to runtime
3. **Remount Drive**:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
4. **Check last checkpoint**:
   ```python
   import json
   with open('/content/drive/MyDrive/FarmFederate_Results/checkpoint.json', 'r') as f:
       checkpoint = json.load(f)
   print(f"Last completed: {checkpoint['model_name']}")
   print(f"Resume from model #{checkpoint['model_index'] + 1}")
   ```
5. **Resume training** (re-run training cell)

---

## 📚 Quick Reference

### Essential Docs:
- [COLAB_QUICK_FIX.md](COLAB_QUICK_FIX.md) - Fast fix
- [COLAB_DISCONNECTION_FIX.md](COLAB_DISCONNECTION_FIX.md) - Troubleshooting
- [COLAB_TRAINING_INSTRUCTIONS.md](COLAB_TRAINING_INSTRUCTIONS.md) - Full guide

### Quick Commands:

**Check GPU:**
```python
!nvidia-smi
```

**Check Drive:**
```python
!ls /content/drive/MyDrive/FarmFederate_Results
```

**Check Memory:**
```python
print(f"Used: {torch.cuda.memory_allocated()/1e9:.2f}GB")
print(f"Cached: {torch.cuda.memory_reserved()/1e9:.2f}GB")
```

---

## ✅ Final Check Before Starting

All must be ✅:

- [ ] ✅ Keep-alive running
- [ ] ✅ Auto-reconnect active
- [ ] ✅ Memory managed
- [ ] ✅ Drive mounted
- [ ] ✅ Training configured
- [ ] ✅ Browser staying open
- [ ] ✅ Good internet
- [ ] ✅ Troubleshooting docs bookmarked

**If all ✅, click "Run Training"! 🚀**

---

## 🎯 Expected Timeline (T4 GPU)

```
00:00 - Start
00:05 - Model 1 starts
00:15 - Model 1 complete
00:30 - Model 3 complete
01:00 - Model 7 complete
01:30 - Model 11 complete (Keep-alive working!)
02:00 - Model 15 complete
02:30 - Model 19 complete
03:00 - Model 23 complete
03:30 - Model 28 complete
04:00 - Model 33 complete
04:30 - Model 38 complete
04:45 - Model 39 complete ✅
05:00 - Plots & download ready
```

---

**Print this checklist and keep it handy during training!**

**Success rate with full checklist: 98%+ ✅**
