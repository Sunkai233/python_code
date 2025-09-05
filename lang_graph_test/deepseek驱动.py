from langchain_deepseek import ChatDeepSeek

llm = ChatDeepSeek(
    model="deepseek-chat",         # 模型名
    temperature=0.7,               # 采样温度（控制生成的随机性）
    max_tokens=1024,               # 最大生成 token 数
    timeout=30.0,                  # 请求超时时间（秒）
    max_retries=2,                 # 请求失败时最多重试次数
    api_key="sk-157abd02156e4718b1132b3ed03fd5ce",        # 可选，默认从环境变量中读取.
)
messages = [
    ("system", "你是一个有帮助的翻译助手，请将用户句子翻译成中文。"),
    ("human", '''
    Podman for Windows
While "containers are Linux," Podman also runs on Mac and Windows, where it provides a native CLI and embeds a guest Linux system to launch your containers. This guest is referred to as a Podman machine and is managed with the podman machine command. On Windows, each Podman machine is backed by a virtualized Windows Subsystem for Linux (WSLv2) distribution. The podman command can be run directly from your Windows PowerShell (or CMD) prompt, where it remotely communicates with the podman service running in the WSL environment. Alternatively, you can access Podman directly from the WSL instance if you prefer a Linux prompt and Linux tooling. In addition to command-line access, Podman also listens for Docker API clients, supporting direct usage of Docker-based tools and programmatic access from your language of choice.

Prerequisites
Since Podman uses WSL, you need a recent release of Windows 10 or Windows 11. On x64, WSL requires build 18362 or later, and 19041 or later is required for arm64 systems. Internally, WSL uses virtualization, so your system must support and have hardware virtualization enabled. If you are running Windows on a VM, you must have a VM that supports nested virtualization.

It is also recommended to install the modern "Windows Terminal," which provides a superior user experience to the standard PowerShell and CMD prompts, as well as a WSL prompt, should you want it.
    '''),
]
#llm.invoke(messages)
for chunk in llm.stream(messages):
    print(chunk)
    #print(chunk.text(), end="")