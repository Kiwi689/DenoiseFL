if [ -n "${ZSH_VERSION:-}" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${(%):-%N}")" && pwd)"
elif [ -n "${BASH_VERSION:-}" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
    SCRIPT_DIR="$(pwd)"
fi

PARENT_DIR="$(dirname "${SCRIPT_DIR}")"
ENV_PATH="${PARENT_DIR}/envs/denoisefl"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PATH}"
EOF

#建议不运行这个脚本 写下面两句 把下面路径改下
#source "$(conda info --base)/etc/profile.d/conda.sh"
#conda activate /root/workspace/kiwi/envs/denoisefl
