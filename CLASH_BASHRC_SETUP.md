# 当前 Clash Bash 配置迁移说明

这份文档记录 5070 Ti 笔记本当前 `~/.bashrc` 中的 Clash 使用方式，供新电脑复现。

## 1. 当前方案的工作方式

当前 Bash 函数本身**不启动 Clash/Mihomo 内核，也不保存订阅**。它依赖：

1. Clash Verge Rev GUI 已经安装并启动；
2. GUI 管理订阅、配置和 Mihomo 内核；
3. Mihomo 提供：
   - mixed port：`127.0.0.1:7897`
   - Unix controller socket：`/tmp/verge/verge-mihomo.sock`
4. Bash 函数负责：
   - 为当前 shell 设置或取消代理环境变量；
   - 通过 Unix socket 查询状态和切换节点；
   - 测试 Google 和出口 IP；
   - 通过 API 开关 TUN。

因此，在新电脑上仅复制下面的 `.bashrc` 代码是不够的，必须先保证 Clash Verge 的 Mihomo 内核正在运行。

## 2. 依赖

Ubuntu/Debian 上至少需要：

```bash
sudo apt update
sudo apt install -y curl python3 iproute2 procps
```

还需要安装 Clash Verge Rev，并在 GUI 中完成：

- 导入自己的订阅；
- 启动 Mihomo 内核；
- 将 mixed port 设置为 `7897`，或者修改函数中的 `PORT`；
- 确认 `/tmp/verge/verge-mihomo.sock` 存在；
- 确认代理组名称和默认节点名称与函数一致。

检查：

```bash
test -S /tmp/verge/verge-mihomo.sock && echo "mihomo socket OK"
curl --unix-socket /tmp/verge/verge-mihomo.sock http://localhost/version
```

## 3. 新电脑上必须检查的四个变量

函数开头有四个与本机配置相关的值：

```bash
local SOCK=/tmp/verge/verge-mihomo.sock
local PORT=7897
local MAIN_GROUP="节点列表"
local DEFAULT_NODE="日本一区·AWS·1.0倍消耗"
```

- `SOCK`：Clash Verge/Mihomo controller socket。
- `PORT`：Clash Verge 的 mixed port。
- `MAIN_GROUP`：订阅中的主代理组名称。
- `DEFAULT_NODE`：执行 `clash on` 时自动选择的节点；新订阅不存在该节点时可以改名，或删除自动选择逻辑。

文档中没有包含订阅 URL、API secret、登录密码或其他私人凭据。

## 4. 追加到 `~/.bashrc` 的函数

把下面完整代码追加到新电脑的 `~/.bashrc`：

```bash
# ===== clash: 命令行管理代理(通过 Clash Verge GUI 内核 mihomo unix socket) =====
# 注意: mihomo 内核的启停/订阅/配置 全部由 Clash Verge GUI 管理,
#       本函数只在终端里设代理环境变量 + 通过 mihomo unix socket API 切节点。
#       使用前请先打开 Clash Verge GUI 并保持内核运行。
#
# 用法:
#   clash on       当前 shell 走代理(经 mihomo 7897)+ 切到默认日本一区AWS节点 + 测 Google 连通性
#   clash off      取消当前 shell 代理(不影响 GUI 内核)
#   clash status   查看内核状态/出口 IP/当前节点
#   clash node     列出可选节点
#   clash node N   切换 节点列表 组到节点 N
#   clash ping     测当前节点到 Google 的连通性(不改 env)
#   clash restart  提示: 内核重启请在 GUI 里操作
#   clash update   提示: 订阅更新请在 GUI 里操作
clash() {
  local SOCK=/tmp/verge/verge-mihomo.sock   # mihomo 内核 unix socket
  local PORT=7897                            # mihomo mixed-port
  local MAIN_GROUP="节点列表"
  local DEFAULT_NODE="日本一区·AWS·1.0倍消耗"

  _clash_api() {
    curl -sS --max-time 5 --unix-socket "$SOCK" "http://localhost/$1" "${@:2}"
  }

  _clash_alive() {
    _clash_api version >/dev/null 2>&1
  }

  _clash_select() {
    local g="$1" n="$2"
    local qg
    qg=$(python3 -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1],safe=''))" "$g")
    _clash_api "proxies/$qg" \
      -X PUT -H "Content-Type: application/json; charset=utf-8" \
      --data-binary "$(python3 -c "import json,sys;print(json.dumps({'name':sys.argv[1]},ensure_ascii=False))" "$n")" \
      -o /dev/null -w "%{http_code}" 2>/dev/null | grep -q 204
  }

  _clash_current() {
    local qg
    qg=$(python3 -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1],safe=''))" "$MAIN_GROUP")
    _clash_api "proxies/$qg" 2>/dev/null \
      | python3 -c "import json,sys;d=json.load(sys.stdin);print(d.get('now','?'))" 2>/dev/null
  }

  _clash_ping_google() {
    local code t ip
    t=$(curl -s --max-time 8 -x "http://127.0.0.1:$PORT" \
        https://www.google.com/generate_204 -o /dev/null -w "%{http_code} %{time_total}" 2>/dev/null)
    code=${t%% *}
    if [ "$code" = "204" ] || [ "$code" = "200" ]; then
      ip=$(curl -s --max-time 8 -x "http://127.0.0.1:$PORT" https://www.cloudflare.com/cdn-cgi/trace 2>/dev/null | grep "^ip=" | cut -d= -f2)
      printf "  Google: \033[32m✓ 通\033[0m  (HTTP %s, %.2fs, 出口 %s)\n" "$code" "${t##* }" "${ip:-?}"
      return 0
    else
      printf "  Google: \033[31m✗ 不通\033[0m  (HTTP %s, %.2fs)\n" "${code:-000}" "${t##* }"
      return 1
    fi
  }

  case "${1:-status}" in
    on|start)
      if ! _clash_alive; then
        echo "clash: mihomo 内核未运行(unix socket 无响应)。请先打开 Clash Verge GUI。"
        return 1
      fi
      if _clash_select "$MAIN_GROUP" "$DEFAULT_NODE"; then
        echo "clash on → 节点: $DEFAULT_NODE"
      else
        echo "clash on: 节点切换失败(订阅里没有「$DEFAULT_NODE」?),仅设代理 env"
      fi
      export http_proxy="http://127.0.0.1:$PORT"
      export https_proxy="http://127.0.0.1:$PORT"
      export all_proxy="socks5://127.0.0.1:$PORT"
      export no_proxy="localhost,127.0.0.1,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,*.local"
      echo "  env: http_proxy/https_proxy/all_proxy → 127.0.0.1:$PORT"
      _clash_ping_google
      ;;

    off|stop)
      unset http_proxy https_proxy all_proxy no_proxy
      echo "clash off (仅取消当前 shell 代理,GUI 内核不受影响)"
      ;;

    status)
      if ! _clash_alive; then
        echo "mihomo: 未运行(请打开 Clash Verge GUI)"
        return 0
      fi
      echo "mihomo: running (unix socket $SOCK, mixed-port $PORT)"
      local ip
      ip=$(curl -sS --max-time 8 -x "http://127.0.0.1:$PORT" https://www.cloudflare.com/cdn-cgi/trace 2>/dev/null | grep "^ip=" | cut -d= -f2)
      echo "出口 IP: ${ip:-?}"
      echo "当前节点: $(_clash_current)"
      echo "代理 env: http=${http_proxy:-未设} all=${all_proxy:-未设}"
      ;;

    node|nodes)
      if ! _clash_alive; then
        echo "clash: 内核未运行,请先打开 GUI"
        return 1
      fi
      if [ -z "${2:-}" ]; then
        echo "节点列表(proxy-group: $MAIN_GROUP):"
        local qg
        qg=$(python3 -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1],safe=''))" "$MAIN_GROUP")
        _clash_api "proxies/$qg" 2>/dev/null \
          | python3 -c "import json,sys;d=json.load(sys.stdin);print('\n'.join(d.get('all',[])))" 2>/dev/null \
          | nl -ba
        echo
        echo "当前: $(_clash_current)"
        echo "用法: clash node <节点名>"
      else
        if _clash_select "$MAIN_GROUP" "$2"; then
          echo "clash → $2"
        else
          echo "切换失败:节点名不存在?"
          return 1
        fi
      fi
      ;;

    ping|test)
      _clash_ping_google
      ;;

    tun)
      if ! _clash_alive; then
        echo "clash: 内核未运行,请先打开 GUI"
        return 1
      fi
      local action="${2:-status}"
      local VERGE_DIR="$HOME/.local/share/io.github.clash-verge-rev.clash-verge-rev"
      case "$action" in
        on)
          if pgrep -f expressvpn-daemon >/dev/null 2>&1; then
            echo -e "clash tun: \033[31mExpressVPN daemon 正在运行\033[0m"
            echo "  ExpressVPN 的策略路由规则(ip rule priority 70-102)会劫持流量,"
            echo "  导致 mihomo TUN(priority 9000+)无法生效。"
            echo "  请先停掉 ExpressVPN daemon:"
            echo "    sudo pkill -f expressvpn-daemon"
            return 1
          fi
          _clash_api configs -X PATCH \
            -H "Content-Type: application/json" \
            -d '{"tun":{"enable":true,"device":"Mihomo","stack":"gVisor","dns-hijack":["any:53"],"auto-route":true,"auto-detect-interface":true,"mtu":1500,"route-exclude-address":["10.0.0.0/8","172.16.0.0/12","192.168.0.0/16"]}}' \
            -o /dev/null -w "%{http_code}" 2>/dev/null | grep -q 204 \
            && echo "clash tun on → TUN 已开启 (设备 Mihomo, gVisor)" \
            || { echo "clash tun on: API 调用失败"; return 1; }
          if grep -q "^enable_tun_mode:" "$VERGE_DIR/verge.yaml" 2>/dev/null; then
            sed -i 's/^enable_tun_mode:.*/enable_tun_mode: true/' "$VERGE_DIR/verge.yaml"
          fi
          sleep 1
          if ip link show Mihomo >/dev/null 2>&1; then
            echo -e "  TUN 接口: \033[32mMihomo UP\033[0m"
            _clash_ping_google
          else
            echo -e "  TUN 接口: \033[31m未创建\033[0m (可能需要 root 权限)"
          fi
          ;;
        off)
          _clash_api configs -X PATCH \
            -H "Content-Type: application/json" \
            -d '{"tun":{"enable":false}}' \
            -o /dev/null -w "%{http_code}" 2>/dev/null | grep -q 204 \
            && echo "clash tun off → TUN 已关闭" \
            || { echo "clash tun off: API 调用失败"; return 1; }
          if grep -q "^enable_tun_mode:" "$VERGE_DIR/verge.yaml" 2>/dev/null; then
            sed -i 's/^enable_tun_mode:.*/enable_tun_mode: false/' "$VERGE_DIR/verge.yaml"
          fi
          ;;
        status)
          local enabled
          enabled=$(curl -s --unix-socket "$SOCK" http://localhost/configs 2>/dev/null \
            | python3 -c "import json,sys;print(json.load(sys.stdin).get('tun',{}).get('enable',False))" 2>/dev/null)
          if [ "$enabled" = "True" ]; then
            echo -e "TUN: \033[32m开启\033[0m"
            ip link show Mihomo 2>/dev/null | head -1 || echo "  接口未创建"
          else
            echo -e "TUN: \033[31m关闭\033[0m"
          fi
          ;;
        *)
          echo "用法: clash tun {on|off|status}"
          return 1
          ;;
      esac
      ;;

    restart)
      echo "clash: 内核启停由 Clash Verge GUI 管理,请在 GUI 里重启内核(或重启 GUI)。"
      ;;

    update)
      echo "clash: 订阅更新由 Clash Verge GUI 管理,请在 GUI「订阅」页点击更新。"
      ;;

    *)
      echo "用法: clash {on|off|status|node|ping|tun|restart|update}"
      echo "      clash tun {on|off|status}"
      echo "      (内核启停/订阅更新请在 Clash Verge GUI 里操作)"
      return 1
      ;;
  esac
}
```

应用：

```bash
source ~/.bashrc
type clash
clash status
```

## 5. 常用命令

```bash
clash status
clash on
clash ping
clash node
clash node "节点完整名称"
clash off
clash tun status
clash tun on
clash tun off
```

说明：

- `clash on/off` 只改变**当前 shell** 的环境变量。
- 新开的 shell 不会自动继承另一个终端里执行的 `clash on`。
- `clash off` 不会关闭 GUI 或 Mihomo 内核。
- `clash tun on/off` 改变系统级 TUN 行为，可能需要 Clash Verge 的服务/root 权限。
- 当前 TUN 配置排除了 RFC1918 内网地址，避免影响局域网和 Tailscale/SSH 所需的本地路径。

## 6. 测试

```bash
clash status
clash on
env | grep -i '_proxy='
curl -I https://www.google.com
curl https://www.cloudflare.com/cdn-cgi/trace | grep '^ip='
clash off
```

## 7. 常见问题

### `mihomo: 未运行`

先启动 Clash Verge GUI，并确认内核处于运行状态：

```bash
ls -l /tmp/verge/verge-mihomo.sock
```

如果新版本 socket 路径变化，应修改 `SOCK`，不要盲目创建同名普通文件。

### `节点切换失败`

订阅中的代理组或节点名称发生了变化。先运行：

```bash
clash node
```

然后修改 `MAIN_GROUP` 和 `DEFAULT_NODE`。

### mixed port 不是 7897

以 Clash Verge GUI 中显示的 mixed port 为准，并修改：

```bash
local PORT=实际端口
```

### 在纯 SSH/headless 服务器上使用

当前函数是 **Clash Verge GUI 专用版本**。如果远端 `hw3090` 没有图形桌面或没有运行 Clash Verge，`/tmp/verge/verge-mihomo.sock` 不会存在。这种情况下应：

1. 将 Mihomo 安装成用户级或 systemd 服务；
2. 使用独立的 `config.yaml`；
3. 开启本地 mixed port 和 controller；
4. 修改函数中的 `SOCK`/API 访问方式；
5. 不要直接复制 Clash Verge 的 GUI 状态目录来代替服务配置。

如果只是让远端训练进程临时使用另一台机器的代理，也可以用 SSH 端口转发，但这需要单独配置，不能直接使用远端的 `127.0.0.1:7897` 指向笔记本。

