import random
import time


def _is_rate_limited(e):
    # transformers 會把 hub 的 HTTP 錯誤包成 ValueError，所以沿著 exception chain 找 429
    while e is not None:
        if "429" in str(e) or "Too Many Requests" in str(e):
            return True
        e = e.__cause__ or e.__context__
    return False


def retry_on_rate_limit(fn, *, attempts=6, base_delay=15.0, max_delay=120.0, log=print):
    """呼叫 fn()；遇到 Hugging Face Hub 429（rate limit）時以指數退避 + 隨機抖動重試。

    每個 rank 啟動時都會向 hub 查 kernel 版本，多個 job 同時啟動（例如 sweep）容易被 rate limit。
    """
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            if not _is_rate_limited(e) or i == attempts - 1:
                raise
            delay = min(max_delay, base_delay * 2 ** i) * random.uniform(0.5, 1.5)
            log(f"WARNING: Hugging Face Hub rate limited (429), retry {i + 1}/{attempts - 1} in {delay:.0f}s")
            time.sleep(delay)
