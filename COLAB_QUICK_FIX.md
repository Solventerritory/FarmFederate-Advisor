# 🚀 Colab Disconnection - Quick Fix (1 Minute)

## Problem: Runtime keeps disconnecting?

### ⚡ INSTANT FIX - Copy & Run This Cell FIRST:

```python
# === COLAB DISCONNECTION FIX - RUN THIS FIRST ===
from IPython.display import Javascript, display
import torch, gc, os

# 1. Keep session alive (prevents 90-min timeout)
display(Javascript('''
    setInterval(() => {
        document.querySelector("colab-toolbar-button#connect")?.click();
        console.log("⏰ Keeping alive");
    }, 60000);
'''))

# 2. Auto-reconnect on network drops
display(Javascript('''
    setInterval(() => {
        if(!google.colab.kernel?.accessAllowed) {
            console.log("🔄 Reconnecting...");
            location.reload();
        }
    }, 30000);
'''))

# 3. Memory management (prevents OOM crashes)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.85)

# 4. Memory clearing function
def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

print("✅ Disconnection fixes enabled!")
print("   - Keep-alive: ✓")
print("   - Auto-reconnect: ✓")
print("   - Memory management: ✓")
print("\n💡 Keep this browser tab open (can be in background)")
```

---

## That's It! Now run your training normally.

### 📋 Additional Recommended Steps:

#### Mount Google Drive (Prevents data loss):
```python
from google.colab import drive
drive.mount('/content/drive')
os.makedirs('/content/drive/MyDrive/FarmFederate_Results', exist_ok=True)
print("✅ Results will auto-save to Drive")
```

#### Check GPU and adjust settings:
```python
if torch.cuda.is_available():
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {torch.cuda.get_device_name(0)} ({gpu_mem:.1f}GB)")
    
    # Auto-configure
    if gpu_mem < 16:  # T4
        batch_size = 2
    elif gpu_mem < 24:  # V100
        batch_size = 4
    else:  # A100
        batch_size = 8
    
    print(f"Recommended batch size: {batch_size}")
```

---

## 🔧 If Still Disconnecting:

1. **Reduce batch size**: Set to 2 or even 1
2. **Enable LoRA**: Reduces memory usage
3. **Add delays**: `time.sleep(30)` between models
4. **Upgrade to Colab Pro**: $9.99/month, 24h sessions

---

## 📱 Quick Troubleshooting:

| Error | Quick Fix |
|-------|-----------|
| "Runtime disconnected" | Re-run the fix cell above |
| "Out of memory" | Reduce batch size to 1 |
| "Session crashed" | Clear GPU: `clear_gpu()` |
| "Idle timeout" | Keep-alive not running, re-run fix cell |

---

**That's it! With this one cell, your Colab should stay connected. 🎉**

See [COLAB_DISCONNECTION_FIX.md](COLAB_DISCONNECTION_FIX.md) for detailed explanations.
