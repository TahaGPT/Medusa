import sqlite3, os

def get_stats(db_path, label):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    try:
        c.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL")
        launches = c.fetchone()[0]
        c.execute("SELECT SUM(end-start) FROM CUPTI_ACTIVITY_KIND_KERNEL")
        gpu_time = c.fetchone()[0] or 0
        c.execute("""
            SELECT s.value, COUNT(*), SUM(k.end-k.start)
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON k.demangledName = s.id
            GROUP BY k.demangledName
            ORDER BY SUM(k.end-k.start) DESC LIMIT 3
        """)
        top_kernels = c.fetchall()
        conn.close()
        return {
            "label": label,
            "launches": launches,
            "gpu_time_ms": gpu_time / 1e6,
            "top_kernels": top_kernels
        }
    except Exception as e:
        conn.close()
        return {"label": label, "error": str(e)}

base_dir = os.path.expanduser("~/Z4RAmode/PDC/Project/Medusa/results")

files = [
    ("baseline_profile.sqlite",          "Baseline (no Medusa)",  94.73, 10.6),
    ("medusa_standard_profile.sqlite",   "Standard Medusa",        8.22, 121.6),
    ("medusa_optimized_profile.sqlite",  "Optimized Medusa",       5.47, 182.7),
]

print("\n" + "="*70)
print("3-WAY COMPARISON RESULTS")
print("="*70)

for fname, label, throughput, latency in files:
    path = os.path.join(base_dir, fname)
    if not os.path.exists(path):
        print(f"\n{label}: MISSING FILE")
        continue
    r = get_stats(path, label)
    if "error" in r:
        print(f"\n{label}: ERROR — {r['error']}")
        continue
    print(f"\n{'─'*50}")
    print(f"  {label}")
    print(f"  Throughput      : {throughput} tok/s")
    print(f"  Latency/token   : {latency} ms")
    print(f"  Kernel launches : {r['launches']:,}")
    print(f"  GPU compute time: {r['gpu_time_ms']:.1f} ms")
    if r['top_kernels']:
        print(f"  Top kernel      : {r['top_kernels'][0][0][:55]}")

print(f"\n{'='*70}")
print("DELTAS vs BASELINE")
print(f"{'='*70}")
baseline_launches = 80219
baseline_throughput = 94.73
rows = [
    ("Standard Medusa",   119951, 8.22),
    ("Optimized Medusa",  155574, 5.47),
]
for label, launches, tp in rows:
    launch_delta = (launches - baseline_launches) / baseline_launches * 100
    tp_delta = (tp - baseline_throughput) / baseline_throughput * 100
    print(f"\n  {label}")
    print(f"  Kernel launches : {launch_delta:+.1f}% vs baseline")
    print(f"  Throughput      : {tp_delta:+.1f}% vs baseline")
