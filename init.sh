#!/usr/bin/env bash
set -euo pipefail

# ---------- 공통 유틸 ----------
color() { # $1=color_name, $2=message
  case "${1:-}" in
    blue)   printf "\e[34m%s\e[0m\n" "$2" ;;
    green)  printf "\e[32m%s\e[0m\n" "$2" ;;
    yellow) printf "\e[33m%s\e[0m\n" "$2" ;;
    red)    printf "\e[31m%s\e[0m\n" "$2" ;;
    *)      printf "%s\n" "$2" ;;
  esac
}

find_python() {
  if command -v python3 >/dev/null 2>&1; then
    echo "python3"
  elif command -v python >/dev/null 2>&1; then
    echo "python"
  else
    color red "Python이 필요합니다. Python을 설치한 뒤 다시 실행하세요."
    exit 1
  fi
}

activate_venv() {
  if [ -f ".venv/bin/activate" ]; then
    # Linux/Mac
    # shellcheck disable=SC1091
    . ".venv/bin/activate"
  elif [ -f ".venv/Scripts/activate" ]; then
    # Windows (Git Bash)
    # shellcheck disable=SC1091
    . ".venv/Scripts/activate"
  else
    color red ".venv 활성화 스크립트를 찾을 수 없습니다."
    exit 1
  fi
}

setup_cuda_env() {
  local os_name cuda_dir
  os_name=$(uname -s 2>/dev/null || echo "UNKNOWN")

  # nvcc 기준 탐색
  if command -v nvcc >/dev/null 2>&1; then
    cuda_dir=$(dirname "$(dirname "$(command -v nvcc)")")
  fi

  # 표준 경로 탐색 (Linux/Mac)
  if [ -z "${cuda_dir:-}" ]; then
    if [ -d "/usr/local/cuda" ]; then
      cuda_dir="/usr/local/cuda"
    else
      # 가장 높은 버전 선택
      local candidate
      candidate=$(ls -d /usr/local/cuda-* 2>/dev/null | sort -V | tail -n1 || true)
      if [ -n "${candidate}" ]; then
        cuda_dir="${candidate}"
      fi
    fi
  fi

  # Windows (Git Bash) 경로 탐색
  if [ -z "${cuda_dir:-}" ] && [ -d "/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA" ]; then
    local win_base win_candidate
    win_base="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA"
    win_candidate=$(ls -d "${win_base}"/v* 2>/dev/null | sort -V | tail -n1 || true)
    if [ -n "${win_candidate}" ]; then
      cuda_dir="${win_candidate}"
    fi
  fi

  if [ -z "${cuda_dir:-}" ]; then
    color yellow "CUDA를 찾을 수 없습니다. CUDA 관련 설정을 건너뛰고 계속 진행합니다."
    return 0
  fi

  export CUDA_HOME="${cuda_dir}"
  export CUDA_PATH="${cuda_dir}"
  export PATH="${cuda_dir}/bin:${PATH}"

  # 라이브러리 경로 설정 (플랫폼별)
  if [ -d "${cuda_dir}/lib64" ]; then
    export LD_LIBRARY_PATH="${cuda_dir}/lib64:${LD_LIBRARY_PATH:-}"
  elif [ -d "${cuda_dir}/lib" ]; then
    export LD_LIBRARY_PATH="${cuda_dir}/lib:${LD_LIBRARY_PATH:-}"
  fi
  # macOS 호환
  if [ -d "${cuda_dir}/lib" ]; then
    export DYLD_LIBRARY_PATH="${cuda_dir}/lib:${DYLD_LIBRARY_PATH:-}"
  fi

  color green "CUDA 설정 완료: ${cuda_dir}"
}

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    color blue "uv가 이미 설치되어 있습니다."
    return 0
  fi
  color blue "uv 설치 중..."
  pip install -U uv >/dev/null 2>&1 || {
    color red "uv 설치에 실패했습니다."
    exit 1
  }
  color green "uv 설치 완료"
}

install_requirements_with_uv() {
  local req_file="requirements.txt"
  if [ -f "${req_file}" ]; then
    color blue "uv로 ${req_file} 설치 중..."
    uv pip install -r "${req_file}"
    color green "요구사항 설치 완료"
  else
    color yellow "${req_file} 파일이 없어 설치를 스킵합니다."
  fi
}

# ---------- 0. .venv 세팅 후 venv 활성화 ----------
if [ ! -d ".venv" ]; then
  if command -v python >/dev/null 2>&1; then
    color blue ".venv 생성 중...(python)"
    python -m venv .venv
  else
    color blue ".venv 생성 중...(python3)"
    if command -v python3 >/dev/null 2>&1; then
      python3 -m venv .venv
    else
      color red "Python이 필요합니다. Python을 설치한 뒤 다시 실행하세요."
      exit 1
    fi
  fi
  color green ".venv 생성 완료"
fi
activate_venv
color green "가상환경 활성화 완료"

# ---------- 1. CUDA 환경 세팅 (없으면 종료) ----------
setup_cuda_env

# ---------- 2. uv 설치 및 requirements 설치 ----------
ensure_uv
install_requirements_with_uv

# ---------- 3. 기존 설정 유지: git / pre-commit ----------
# git 설정
git config --global commit.template ./.commit_template || true
git config --global core.editor "code --wait" || true
color blue "Fin git config"

# pre-commit 설정
pip install -U pre-commit >/dev/null 2>&1 || true
pre-commit autoupdate || true
pre-commit install || true
color blue "Fin pre-commit"
