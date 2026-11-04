# prima.cpp 功能移植狀態：程式碼位置對照

本文件對應〈prima.cpp 分散式推論移植至 llama.cpp — 工作進度與程式修改報告〉第七章「總結：prima.cpp 功能移植狀態」，
逐列標出每項功能的程式碼位置：

- **原版**：`prima.cpp/` 子專案（commit `8894dc3` 加入的原始 prima.cpp 程式碼）
- **移植**：本分支 `claude/sharp-keller-7f4k7h`（截至 commit `4b78c9a`）中對應的實作

行號以撰寫當下的版本為準，格式為 `檔案:行號`。

## 總覽

| # | 功能 | 狀態 | 原版主要位置 | 移植主要位置 |
|---|------|------|--------------|--------------|
| 1 | 環狀管線分散式推論 | 已完成 | `prima.cpp/src/llama.cpp:18229` `llama_decode_internal` | `src/llama-context.cpp:1743` `pp_process_ubatch`、`src/llama-prima.cpp` |
| 2 | 各節點只載入自己的層 | 已完成 | `prima.cpp/src/llama.cpp:3838` `this_layer_is_mine` | `src/llama-model.cpp:2424`、`2515` `pp_owns_layer` |
| 3 | 手動指定層窗口 | 已完成 | `prima.cpp/common/arg.cpp:708`（`-lw`） | `common/arg.cpp:4844`、`src/llama-prima.cpp:37` |
| 4 | 多循環（多圈）推論 | 已完成 | `prima.cpp/src/llama.cpp:16941` `map_layer_to_subgf_id` | `src/llama-prima.cpp:37` `llama_prima_make_windows` |
| 5 | KV cache 操作同步 | 已完成 | `prima.cpp/src/llama.cpp:22610`–`22740` `llama_send_kv_cache_*` | `src/llama-context.cpp:4800`–`4880`、`src/llama-prima.cpp:83`、`495` |
| 6 | 裝置效能分析 | 已完成 | `prima.cpp/common/profiler.cpp`、`prima.cpp/src/llama.cpp:3621` | `common/prima-profiler.cpp:607` `prima_profile_device` |
| 7 | 自動層分配（HiGHS 求解） | 部分完成 | `prima.cpp/common/common.cpp:860` `assign_layers_to_device` | `common/prima.cpp:84` `assign_layers_to_devices` |
| 8 | 裝置篩選與轉發節點 | 已完成 | `prima.cpp/common/common.cpp:1595`、`prima.cpp/src/llama.cpp:20684` | `common/prima.cpp:549`、`657` |
| 9 | 權重預取 | 已完成 | `prima.cpp/src/llama.cpp:18152` `manage_graph_tensors` | `src/llama-prima.cpp:423`、`456`、`src/llama-context.cpp:1685`、`1726` |
| 10 | 輸出層放置 | 部分完成 | `prima.cpp/src/llama.cpp:7564`、`7612` | `src/llama-model.cpp:2496` `pp_assign_output` |
| 11 | GPU 記憶體上限（`--gpu-mem`） | 已完成 | `prima.cpp/common/arg.cpp:763`、`prima.cpp/src/llama.cpp:3677` | `common/arg.cpp:4942`、`common/prima-profiler.cpp:669` |
| 12 | 主節點優先權（`--master-priority`） | 已完成 | `prima.cpp/common/arg.cpp:787`、`prima.cpp/common/common.cpp:1280` | `common/arg.cpp:4935`、`common/prima.cpp:333` |
| 13 | 分散式 llama-server | 已完成 | `prima.cpp/examples/server/server.cpp:716`、`prima.cpp/src/llama.cpp:18232`（`server_mode`） | `tools/server/server.cpp:574`、`tools/server/server-context.cpp:1573` |
| 14 | 投機解碼 | 部分完成 | `prima.cpp/examples/speculative/speculative.cpp:73` | `common/speculative.cpp:2662`、`common/common.cpp:1264` |
| 15 | llama-profile-tool | 已完成 | `prima.cpp/tools/profile_tool.cpp` | `tools/profile-tool/profile-tool.cpp` |
| 16 | GGML_OP_READ | 部分完成 | `prima.cpp/ggml/src/ggml-cuda/read.cu`、`prima.cpp/ggml/src/ggml-metal.metal:3278` | `ggml/src/ggml.c:2577`、`ggml/src/ggml-cpu/ops.cpp:1588`、`ggml/src/ggml-cuda/read.cu` |
| 17 | Makefile 建置選項轉 CMake | 已完成 | `prima.cpp/Makefile:273`、`280` | `CMakeLists.txt:147`、`231`–`354` |
| 18 | 支援模型架構 | 已完成 | `prima.cpp/src/llama.cpp:11000` `build_llama`、`12736` `build_qwen2` | `src/models/{llama,qwen2,qwen3,granite}.cpp`、`src/llama-model.cpp:2457` |
| 19 | 支援工具 | 已完成 | `prima.cpp/examples/main/main.cpp:146` | `tools/completion/completion.cpp:177` 等 |
| 20 | 嵌入向量（embedding） | 未實作 | `prima.cpp/src/llama.cpp:18604` | — |
| 21 | 推論狀態儲存與載入 | 未實作 | `prima.cpp/src/llama.cpp:22771`、`22780` | `src/llama-context.cpp:3997`（回報錯誤） |
| 22 | 獨立 signal port 與 master IP | 未實作 | `prima.cpp/src/llama.cpp:20492`、`20887` | `common/arg.cpp:4878`、`4899`（僅解析） |
| 23 | 其他模型架構 | 未實作 | `prima.cpp/src/llama.cpp:17003`（`llama_build_graph` 的架構檢查） | `src/llama-model.cpp:2457`–`2466`（拒絕） |
| 24 | Vulkan 等其他 GPU 後端 | 未實作 | `prima.cpp/common/profiler.cpp:1343` 等 | `common/prima-profiler.cpp:657`–`661` |
| 25 | Windows 與 Android 實機測試 | 部分完成 | `prima.cpp/common/profiler.cpp:496`、`1491` | `common/prima-profiler.cpp:238`–`245`、`289`、`416` |

---

## 1. 環狀管線分散式推論

多台機器依層窗口分工，以 ZeroMQ 傳遞 hidden state。狀態：**已完成**。

| 原版 | 說明 |
|------|------|
| `prima.cpp/src/llama.cpp:18229` `llama_decode_internal()` | 分散式解碼主流程：收、算、送 |
| `prima.cpp/src/llama.cpp:18500`–`18565` | 逐子圖接收 (`llama_recv_tensors`)、計算、送出 (`llama_send_tensors`) |
| `prima.cpp/src/llama.cpp:17870` / `17920` | `llama_send_meta()` / `llama_recv_meta()` |
| `prima.cpp/src/llama.cpp:18031` / `18054` | `llama_send_tensors()` / `llama_recv_tensors()` |
| `prima.cpp/src/llama.cpp:20492` | `llama_init_sockets()` 建立 ZeroMQ 連線 |
| `prima.cpp/src/llama.cpp:3446`–`3449` | context 內的 `send/recv/master/signal_socket` |
| `prima.cpp/src/network-utils.cpp` | 網路工具 |

| 移植 | 說明 |
|------|------|
| `src/llama-prima.h:19`–`130` | 窗口、訊息種類、KV 操作、傳送資料與 `llama_prima_net` 類別定義 |
| `src/llama-prima.cpp:148`–`280` | ZeroMQ 環狀傳輸實作（`send_parts`、`recv_parts`、`recv`、`forward`） |
| `src/llama-prima.cpp:282`–`306` | 沒有 ZeroMQ 時的替代實作（執行時回報錯誤） |
| `src/llama-prima.cpp:310`–`366` | `send_data`、`send_ops`、`send_stop`、`send_ctl`、`recv_ctl` |
| `src/llama-context.cpp:518`–`548` | 建立 context 時建立或沿用環狀連線 |
| `src/llama-context.cpp:638`–`646` | 結束時主節點送出停止訊號 |
| `src/llama-context.cpp:1664` / `1670` | `pp_set_window()`、`pp_get_hidden()` |
| `src/llama-context.cpp:1743` | `pp_process_ubatch()`：主節點嵌入→送出→接收→計算→logits |
| `src/llama-context.cpp:1868` | `pp_worker_run()`：worker 主迴圈（資料、KV 操作、停止） |
| `src/llama-context.cpp:2362`–`2390` | `decode()` 改走分散式流程 |
| `src/llama-context.cpp:5125` | 公開 API `llama_prima_worker_run()` |
| `src/llama-graph.cpp:1547`–`1548`、`2642`、`2823` | 計算圖層範圍、hidden state 輸入、`build_pp_out()` 輸出 |
| `include/llama.h:358`–`364`、`436`–`442`、`820`–`848` | 公開 API 與參數 |

## 2. 各節點只載入自己的層

權重與 KV cache 只建立本機負責的部分。狀態：**已完成**。

| 原版 | 說明 |
|------|------|
| `prima.cpp/src/llama.cpp:3838` `this_layer_is_mine()` | 判斷某層是否屬於本機 |
| `prima.cpp/src/llama.cpp:3889` `llama_kv_cache_init()`（`3921`、`3959`） | KV cache 只建立本機層 |
| `prima.cpp/src/llama.cpp:7554` `llm_load_tensors_impl()`（`7585`–`7594`） | 只載入本機層的張量 |
| `prima.cpp/src/llama.cpp:9536` `llm_load_tensors()` | 載入入口 |

| 移植 | 說明 |
|------|------|
| `src/llama-model.cpp:2424`–`2440` | `create_tensor()`：略過非本機層，worker 不載入嵌入層與輸出層 |
| `src/llama-model.cpp:2444` `pp_init()` | 計算層分配 |
| `src/llama-model.cpp:2515` `pp_owns_layer()` | 判斷層是否屬於本機 |
| `src/llama-model.cpp:3388`–`3392` | `create_memory()`：KV cache 只建立本機層 |
| `src/llama-model.cpp:1929`–`1930` | 不預讀整個模型檔 |
| `src/llama-model-loader.cpp:1188`–`1200`、`1315` | 非本機張量只記錄除錯訊息並正確計數 |

## 3. 手動指定層窗口

`-lw` 指定每台機器的層數，`-ngl` 指定 GPU 層數。狀態：**已完成**。

| 原版 | 說明 |
|------|------|
| `prima.cpp/common/arg.cpp:680`、`687`、`708` | `--world`、`--rank`、`-lw` 參數 |
| `prima.cpp/src/llama.cpp:2604` | `n_layer_window[32]` 欄位 |
| `prima.cpp/src/llama.cpp:7592`–`7612` | 依窗口分配 GPU 層 |
| `prima.cpp/examples/main/main.cpp:146`–`160` | 檢查 `-lw` 設定合法性 |

| 移植 | 說明 |
|------|------|
| `common/arg.cpp:4830`、`4837`、`4844` | `--world`、`--rank`、`-lw` 參數 |
| `src/llama-prima.cpp:37` `llama_prima_make_windows()` | 計算窗口並檢查設定 |
| `src/llama-model.cpp:1689`–`1705` | 每個窗口的最後 n 層放 GPU |
| `common/common.cpp:1757`–`1763` | 把參數傳給 `llama_model_params` |
| `tools/completion/completion.cpp:98`–`115` | 檢查 `-lw` 設定合法性 |

## 4. 多循環（多圈）推論

窗口總和小於層數時資料繞環多圈。狀態：**已完成**。

| 原版 | 說明 |
|------|------|
| `prima.cpp/src/llama.cpp:3855`–`3880` | 計算 cycle 大小與本機層偏移 |
| `prima.cpp/src/llama.cpp:16941` `map_layer_to_subgf_id()` | 層對應到子圖 |
| `prima.cpp/common/arg.cpp:774` | `n_cycles` 參數 |

| 移植 | 說明 |
|------|------|
| `src/llama-prima.cpp:37`–`80` `llama_prima_make_windows()` | 產生多圈窗口，檢查能否整除層數 |
| `src/llama-model.cpp:2487` `pp_window_of_layer()` | 查詢層所在窗口 |
| `src/llama-context.cpp:1743`–`1866` | 依窗口序列逐圈處理 |
| `common/arg.cpp:4953` | `n_cycles` 參數 |

## 5. KV cache 操作同步

清除、刪除、複製、平移等操作轉送給 worker。狀態：**已完成**。

| 原版 | 說明 |
|------|------|
| `prima.cpp/src/llama.cpp:126` `kv_cache_op()` | worker 端套用操作 |
| `prima.cpp/src/llama.cpp:18329`–`18386` | 解碼時處理收到的 KV 操作並轉送 |
| `prima.cpp/src/llama.cpp:22610`、`22629`、`22653`、`22675`、`22698`、`22724` | `llama_send_kv_cache_clear/seq_rm/seq_cp/seq_keep/seq_add/seq_div` |

| 移植 | 說明 |
|------|------|
| `src/llama-prima.cpp:83` `llama_prima_op_apply()` | worker 端套用操作 |
| `src/llama-prima.cpp:485`–`508` | `llama_prima_mem_register/unregister/record` 記錄操作 |
| `src/llama-prima.cpp:410` `flush_ops()` | 把累積的操作送出 |
| `src/llama-context.cpp:4800`–`4876` | `llama_memory_clear/seq_rm/seq_cp/seq_keep/seq_add/seq_div` 中記錄操作 |
| `src/llama-context.cpp:2362`–`2365` | 解碼前先送出操作 |

## 6. 裝置效能分析

CPU／GPU 運算速度、記憶體、頻寬、磁碟速度。狀態：**已完成**（CUDA／Metal 部分未在實機驗證）。

| 原版 | 說明 |
|------|------|
| `prima.cpp/src/llama.cpp:3621` `llama_profile_device()` | 量測入口 |
| `prima.cpp/src/llama.cpp:22144` `llama_model_n_flops()`、`21930` `llama_model_compute_buf_size()`、`22098` `llama_total_kv_size()` | 模型計算量、緩衝區與 KV 大小 |
| `prima.cpp/common/profiler.cpp:165` `device_flops()` | CPU／GPU 矩陣乘法速度 |
| `prima.cpp/common/profiler.cpp:685`、`799` | 記憶體、swap 容量 |
| `prima.cpp/common/profiler.cpp:1066`–`1074` | 磁碟（fio）與記憶體頻寬 |
| `prima.cpp/common/profiler.cpp:1216` `device_mem_copy()` | KV 複製時間 |
| `prima.cpp/common/profiler.cpp:1712` `device_print_props()` | 印出量測表 |
| `prima.cpp/common/profiler.cpp:2393` / `2634` | `serialize()` / `deserialize()` |

| 移植 | 說明 |
|------|------|
| `common/prima.h:17` / `37` | `prima_model_stats`、`prima_device_info` |
| `common/prima-profiler.cpp:59` `prima_model_stats_from_file()` | 從模型檔讀出計算量與資料量（支援分割檔） |
| `common/prima-profiler.cpp:286`–`378` | 記憶體、swap、可置換記憶體 |
| `common/prima-profiler.cpp:380` / `412` | 記憶體頻寬、磁碟速度 |
| `common/prima-profiler.cpp:564` / `594` | `device_gflops()`、`device_kv_cpy_ms()` |
| `common/prima-profiler.cpp:607` `prima_profile_device()` | 量測入口 |
| `common/prima-profiler.cpp:684` `prima_compute_buf_size()` | 估算計算緩衝區 |
| `common/prima-profiler.cpp:710` `prima_print_devices()` | 印出量測表 |

## 7. 自動層分配（HiGHS 求解）

整數線性規劃求最佳分層與循環數。狀態：**部分完成**（HiGHS 路徑尚未實測，無 HiGHS 時改用記憶體比例分配）。

| 原版 | 說明 |
|------|------|
| `prima.cpp/common/common.cpp:860` `assign_layers_to_device()` | 求解流程（M1–M4 分組、迭代） |
| `prima.cpp/common/common.cpp:889` | `#if defined(USE_HIGHS)` HiGHS 求解 |
| `prima.cpp/common/common.cpp:1054` | 選出可用的循環數 `valid_k` |
| `prima.cpp/common/common.cpp:1446`–`1476` | 判斷 GPU／CPU 過載與弱裝置 |

| 移植 | 說明 |
|------|------|
| `common/prima.cpp:56` `init_windows()` | 依記憶體預算初始化 |
| `common/prima.cpp:84` `assign_layers_to_devices()` | 求解流程 |
| `common/prima.cpp:108`–`515` | HiGHS 求解（`Highs_mipCall` 位於 `396`） |
| `common/prima.cpp:156`–`163` | `valid_k` 循環數 |
| `common/prima.cpp:516`–`547` | 無 HiGHS 時依記憶體／顯示記憶體比例分配 |

## 8. 裝置篩選與轉發節點

移除只分到 1 層的裝置，必要時保留為轉發節點。狀態：**已完成**。

| 原版 | 說明 |
|------|------|
| `prima.cpp/common/common.cpp:1595` `assign_layers_and_select_devices()` | 移除只分到 1 層的裝置並重新求解 |
| `prima.cpp/common/common.cpp:1752`–`1801` | 啟動參數廣播、收集裝置資料、廣播分層、重建拓撲 |
| `prima.cpp/src/llama.cpp:20590`、`20649`、`20851` | `llama_bcast_startup_args`、`llama_bcast_layer_setup`、`llama_recv_layer_setup` |
| `prima.cpp/src/llama.cpp:20529` / `20563` | `llama_gather_device_info` / `llama_send_device_info` |
| `prima.cpp/src/llama.cpp:20684` `llama_rebuild_topo()` | 重建拓撲、標記轉發節點 |
| `prima.cpp/src/llama.cpp:20832` `llama_forward_messages()` | 轉發節點迴圈 |
| `prima.cpp/src/llama.cpp:20471` `try_connect()` | 連線測試 |

| 移植 | 說明 |
|------|------|
| `common/prima.cpp:549` `prima_assign_layers()` | 移除裝置並重新求解（`573` 印出移除訊息） |
| `common/prima.cpp:657` `prima_topo_step()` | 重建拓撲、標記轉發節點 |
| `common/prima.cpp:717` `prima_setup()` | 環設定流程（`731` 啟動參數、`785` 收集裝置資料、`816` 廣播分層、`842` 重建拓撲、`875` 重新編號） |
| `src/llama-prima.cpp:214` `forward()` | 轉發節點迴圈 |
| `src/llama-prima.cpp:368` `port_open()` | TCP 連線測試 |
| `src/llama-prima.cpp:510`–`572` | `llama_prima_net_*`、`llama_prima_port_open` 公開 API |
| `common/common.cpp:1276`–`1281` | 載入模型前呼叫 `prima_setup()` |

## 9. 權重預取（`--prefetch`、`--force`）

算完窗口後預讀下一窗口權重。狀態：**已完成**（重新打包的 CPU 權重無法受益）。

| 原版 | 說明 |
|------|------|
| `prima.cpp/common/arg.cpp:756` / `780` | `--prefetch` / `--force` |
| `prima.cpp/src/llama.cpp:18152` `manage_graph_tensors()` | 頁對齊合併、`posix_madvise`（`18207`） |
| `prima.cpp/src/llama.cpp:18567`–`18573` | 送出結果後預取下一個窗口 |

| 移植 | 說明 |
|------|------|
| `common/arg.cpp:4906` / `4928` | `--prefetch` / `--force` |
| `src/llama-prima.cpp:423` `llama_prima_make_segments()` | 頁對齊並合併範圍 |
| `src/llama-prima.cpp:456` `llama_prima_prefetch()` | 請作業系統預讀，可強制讀入頁面 |
| `src/llama-context.cpp:1685` `pp_init_prefetch()` | 建立各窗口的預取範圍 |
| `src/llama-context.cpp:1726` `pp_prefetch_after()` | 窗口送出後預取下一窗口（呼叫點 `1806`、`1853`） |

## 10. 輸出層放置

`--keep-out-in-cuda`、`--keep-out-in-metal`。狀態：**部分完成**（GPU 實機未驗證）。

| 原版 | 說明 |
|------|------|
| `prima.cpp/common/arg.cpp:799`（註解掉）/ `809` | `--keep-out-in-metal` / `--keep-out-in-cuda` |
| `prima.cpp/include/llama.h:327`–`328`、`342`–`343` | 參數欄位 |
| `prima.cpp/src/llama.cpp:7564`–`7612` | 載入時決定輸出層放置 |
| `prima.cpp/src/llama.cpp:21381` | 輸出層子圖的後端選擇 |
| `prima.cpp/common/common.cpp:1241`、`1578` | 求解時考慮輸出層在 Metal |

| 移植 | 說明 |
|------|------|
| `common/arg.cpp:4913` / `4920` | `--keep-out-in-cuda` / `--keep-out-in-metal` |
| `include/llama.h:360`–`361` | 參數欄位 |
| `src/llama-model.cpp:2496` `pp_assign_output()` | 依參數把輸出層放 GPU 或 CPU（呼叫點 `1729`–`1730`） |

## 11. GPU 記憶體上限（`--gpu-mem`）

限制分層時可用的顯示記憶體。狀態：**已完成**。

| 原版 | 說明 |
|------|------|
| `prima.cpp/common/arg.cpp:763` | `-gm`、`--gpu-mem` |
| `prima.cpp/src/llama.cpp:3677` | 在效能分析中限制 `memory_free` |

| 移植 | 說明 |
|------|------|
| `common/arg.cpp:4942` | `-gm`、`--gpu-mem` |
| `common/prima-profiler.cpp:669` | 限制 `gpu_mem_free` |
| `common/prima.cpp:134`、`291`–`303`、`519`、`536` | 求解時作為顯示記憶體預算 |

## 12. 主節點優先權（`--master-priority`）

調整工作優先分配給主節點的程度。狀態：**已完成**。

| 原版 | 說明 |
|------|------|
| `prima.cpp/common/arg.cpp:787` | `--master-priority` |
| `prima.cpp/common/common.cpp:1280` | 套用於求解成本 |

| 移植 | 說明 |
|------|------|
| `common/arg.cpp:4935` | `--master-priority` |
| `common/prima.cpp:333` | 套用於求解成本 |

## 13. 分散式 llama-server

多使用者、多 slot、動態 batching。狀態：**已完成**。

| 原版 | 說明 |
|------|------|
| `prima.cpp/src/llama.cpp:18232`、`18386` | `llama_decode_internal(..., server_mode)`：對齊 seq id |
| `prima.cpp/examples/server/server.cpp:715`–`732` | 草稿模型不切分、只在主節點 |

| 移植 | 說明 |
|------|------|
| `tools/server/server.cpp:574` `server_run_prima_worker()` | 非主節點只載入自己的層並進入 worker 迴圈 |
| `tools/server/server.cpp:613`–`615` | 依 rank 分流 |
| `tools/server/server-context.cpp:1573`–`1581` | 關閉 prompt cache、閒置 slot 快取與 context checkpoint |
| `common/common.cpp:1289` | 多節點時略過依記憶體自動調整參數 |

## 14. 投機解碼

草稿模型與 n-gram。狀態：**部分完成**（MTP、Eagle3 等需讀取 hidden state 的類型在多節點下不支援）。

| 原版 | 說明 |
|------|------|
| `prima.cpp/examples/speculative/speculative.cpp:73`–`76` | 草稿模型 `n_world = 1`、`rank = 0` |
| `prima.cpp/examples/server/server.cpp:715`–`732` | server 的草稿模型 |

| 移植 | 說明 |
|------|------|
| `common/speculative.cpp:2662`–`2663` | 草稿模型不切分 |
| `common/common.cpp:1264`–`1274` | 多節點時拒絕 MTP、Eagle3、DFlash、DSpark |
| `examples/speculative/speculative.cpp:93`–`95`、`111`–`112` | worker 迴圈、草稿模型不切分 |
| `examples/speculative-simple/speculative-simple.cpp:52`–`54` | worker 迴圈 |
| `tools/server/server.cpp:576`–`578` | worker 不載入草稿模型 |

## 15. llama-profile-tool

單機裝置效能分析工具。狀態：**已完成**。

| 原版 | 說明 |
|------|------|
| `prima.cpp/tools/profile_tool.cpp:16`–`58` | `main()`，呼叫 `llama_profile_device` 與 `device_print_props` |
| `prima.cpp/Makefile:1552` | `profile-tool` 目標 |

| 移植 | 說明 |
|------|------|
| `tools/profile-tool/profile-tool.cpp:19`–`36` | `main()`，呼叫 `prima_profile_device` 與 `prima_print_devices` |
| `tools/profile-tool/CMakeLists.txt` | 建置目標 |
| `tools/CMakeLists.txt:48`–`50` | 加入子目錄 |

## 16. GGML_OP_READ 記憶體讀取運算

CPU、CUDA、Metal 實作。狀態：**部分完成**（CPU 已驗證；CUDA、Metal 未能編譯測試）。

| 原版 | 說明 |
|------|------|
| `prima.cpp/ggml/include/ggml.h:530` | `GGML_OP_READ` |
| `prima.cpp/ggml/src/ggml.c:17495` | CPU 端為 no-op |
| `prima.cpp/ggml/src/ggml-cuda/read.cu:4`–`26`、`prima.cpp/ggml/src/ggml-cuda.cu:2149` | CUDA kernel 與分派 |
| `prima.cpp/ggml/src/ggml-metal.metal:3278` `kernel_read_vram` | Metal kernel |
| `prima.cpp/ggml/src/ggml-metal.m:720`、`832`、`3004` | Metal 註冊與分派 |
| `prima.cpp/common/profiler.cpp:1122` `device_read_vram_bw()` | 用於量測顯示記憶體讀取頻寬 |

| 移植 | 說明 |
|------|------|
| `ggml/include/ggml.h:605` | `GGML_OP_READ` |
| `ggml/src/ggml.c:1112`、`2577` `ggml_read()` | 運算名稱與建構函式 |
| `ggml/src/ggml-cpu/ops.cpp:1588` `ggml_compute_forward_read()` | CPU 多執行緒實作 |
| `ggml/src/ggml-cpu/ggml-cpu.c:2157`、`2508` | CPU 分派 |
| `ggml/src/ggml-cuda/read.cu`、`ggml/src/ggml-cuda/ggml-cuda.cu:2268`、`5902` | CUDA 實作與分派 |
| `ggml/src/ggml-metal/kernels/reduce.metal`、`ggml/src/ggml-metal/ggml-metal-ops.cpp:335`、`1009`、`ggml/src/ggml-metal/ggml-metal-device.cpp:357` | Metal 實作與分派 |
| `ggml/include/ggml-rpc.h:11`、`14` | RPC 協定版本 |
| `tests/test-backend-ops.cpp:3012` | 後端測試案例 |
| `common/prima-profiler.cpp:582`、`674`、`742` | `device_read_bw()` 量測並顯示 |

## 17. Makefile 建置選項轉 CMake

ZeroMQ、HiGHS、prima 建置目標。狀態：**已完成**（HiGHS 連結未實測）。

| 原版 | 說明 |
|------|------|
| `prima.cpp/Makefile:7` | `BUILD_TARGETS` 中的 `profile-tool` |
| `prima.cpp/Makefile:220` | `default` 目標 |
| `prima.cpp/Makefile:273` | `-lzmq` |
| `prima.cpp/Makefile:280`–`288` | `USE_HIGHS=1` |

| 移植 | 說明 |
|------|------|
| `CMakeLists.txt:147`–`148` | `LLAMA_PRIMA`、`LLAMA_PRIMA_HIGHS` 選項 |
| `CMakeLists.txt:231`–`275` | ZeroMQ 尋找（pkg-config 或直接搜尋） |
| `CMakeLists.txt:277`–`297` | HiGHS 尋找 |
| `CMakeLists.txt:300` `llama_prima_link_deps()` | 共用連結函式 |
| `CMakeLists.txt:347`–`354` | `prima` 建置目標 |
| `src/CMakeLists.txt`、`common/CMakeLists.txt`、`tools/CMakeLists.txt:48`–`50` | 子目錄的連結與原始檔 |

## 18. 支援模型架構

llama、qwen2、qwen3、granite、granite_moe。狀態：**已完成**。

| 原版 | 說明 |
|------|------|
| `prima.cpp/src/llama.cpp:11000` `build_llama()` | llama（也供 granite、granite_moe 使用） |
| `prima.cpp/src/llama.cpp:12736` `build_qwen2()` | qwen2 |
| `prima.cpp/src/llama.cpp:17003` | `llama_build_graph()` 中的 `GGML_ASSERT` 只允許 LLAMA 與 QWEN2（原版實際上會擋下 granite） |
| `prima.cpp/src/llama.cpp:17005`–`17010`、`17056` | `llama_build_graph()` 分派 |

| 移植 | 說明 |
|------|------|
| `src/llama-model.cpp:2457`–`2466` | 檢查架構 |
| `src/models/llama.cpp:110`–`131`、`236` | llama 只跑指定層範圍 |
| `src/models/qwen2.cpp:64`–`76`、`142` | qwen2 |
| `src/models/qwen3.cpp:59`、`81`–`93`、`167` | qwen3 |
| `src/models/granite.cpp:138`–`152`、`194` | granite（granite_moe 共用） |

## 19. 支援工具

completion、server、cli、speculative、speculative-simple。狀態：**已完成**。

| 原版 | 說明 |
|------|------|
| `prima.cpp/examples/main/main.cpp:146`–`160`、`299`、`355`、`460` | llama-cli（main）依 rank 分流 |
| `prima.cpp/examples/server/server.cpp` | llama-server |
| `prima.cpp/examples/speculative/speculative.cpp:73`–`76` | llama-speculative |

| 移植 | 說明 |
|------|------|
| `tools/completion/completion.cpp:98`–`115`、`164`–`177` | llama-completion 非主節點執行 worker 迴圈 |
| `tools/server/server.cpp:574`、`613` | llama-server |
| `examples/speculative/speculative.cpp:93` | llama-speculative |
| `examples/speculative-simple/speculative-simple.cpp:52` | llama-speculative-simple |

## 20. 嵌入向量（embedding）

分散式模式下輸出 embedding。狀態：**未實作**。

| 原版 | 說明 |
|------|------|
| `prima.cpp/src/llama.cpp:18279`、`18476`–`18487`、`18604`–`18630` | `llama_decode_internal()` 中 embedding 的計算與取出 |

| 移植 | 說明 |
|------|------|
| — | 尚未移植 |

## 21. 推論狀態儲存與載入

session 檔、server prompt cache、slot 儲存／還原。狀態：**未實作**（分散式模式下會明確回報錯誤）。

| 原版 | 說明 |
|------|------|
| `prima.cpp/src/llama.cpp:22771`、`22776` | `llama_state_load_file` / `llama_state_save_file` |
| `prima.cpp/src/llama.cpp:22780` | `struct llama_data_write` |

| 移植 | 說明 |
|------|------|
| `src/llama-context.cpp:3997`、`4021`、`4051`、`4063` | `state_*_data()` 在分散式模式下拋出錯誤 |
| `tools/server/server-context.cpp:1573`–`1581` | 關閉 server prompt cache 與 checkpoint |

## 22. 獨立 signal port 與 master IP

原版 `--signal-port`、`--master` 參數。狀態：**未實作**（參數可解析但目前未使用，控制訊息改走資料環）。

| 原版 | 說明 |
|------|------|
| `prima.cpp/common/arg.cpp:701` / `742` | `--signal-port` / `--master` |
| `prima.cpp/src/llama.cpp:3441`–`3444` | `master_ip`、`signal_port` 欄位 |
| `prima.cpp/src/llama.cpp:20511`–`20512` | `llama_init_sockets()` 建立 master 與 signal 端點 |
| `prima.cpp/src/llama.cpp:20887`–`20896` | `llama_free_sockets()` 透過 signal port 送出停止訊號 |

| 移植 | 說明 |
|------|------|
| `common/arg.cpp:4878` / `4899` | 參數只解析，未使用 |
| `src/llama-prima.cpp:338`、`344` | 停止與控制訊息改走資料環（`send_stop`、`send_ctl`） |

## 23. 其他模型架構

上述五種以外的架構。狀態：**未實作**（分散式模式下會拒絕）。

| 原版 | 說明 |
|------|------|
| `prima.cpp/src/llama.cpp:17003` | `llama_build_graph()` 以 `GGML_ASSERT` 拒絕 LLAMA、QWEN2 以外的架構 |
| `prima.cpp/src/llama.cpp:17012` 起 | 其他架構只產生單一子圖（`result.push_back(...)`），不支援切分 |

| 移植 | 說明 |
|------|------|
| `src/llama-model.cpp:2463`–`2473` | 不支援的架構、deepstack 與 nextn/MTP 模型一律拒絕 |

## 24. Vulkan 等其他 GPU 後端

求解器只把 CUDA／ROCm／MUSA 與 Metal 視為 GPU。狀態：**未實作**（與原版相同限制）。

| 原版 | 說明 |
|------|------|
| `prima.cpp/common/profiler.cpp:308`、`321` | 只量測 Metal 與 CUDA |
| `prima.cpp/common/profiler.cpp:1335`–`1359` | `device_has_*` 偵測（Vulkan 等僅偵測，不參與分層） |
| `prima.cpp/common/common.cpp:850` `get_backend_type()` | 後端類型判斷 |

| 移植 | 說明 |
|------|------|
| `common/prima-profiler.cpp:657`–`661` | 只把 CUDA、ROCm、MUSA 與 Metal 視為 GPU，其他後端印出警告 |

## 25. Windows 與 Android 實機測試

跨平台叢集。狀態：**部分完成**（程式有平台判斷，但未在這些平台實測）。

| 原版 | 說明 |
|------|------|
| `prima.cpp/common/profiler.cpp:496` `device_host_physical_memory()` | 各平台記憶體查詢 |
| `prima.cpp/common/profiler.cpp:1491` `device_termux_swappable_memory()` | Android（Termux） |
| `prima.cpp/common/profiler.cpp:1549` `device_macos_swappable_memory()` | macOS |

| 移植 | 說明 |
|------|------|
| `common/prima-profiler.cpp:26`、`42` | 平台相依標頭 |
| `common/prima-profiler.cpp:238`–`245` `device_os()` | Windows／Android／macOS 判斷 |
| `common/prima-profiler.cpp:289`–`360` | 各平台記憶體與 swap 查詢 |
| `common/prima-profiler.cpp:416` | Windows 磁碟量測分支 |
