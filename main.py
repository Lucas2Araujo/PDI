"""
main.py — Ponto de Entrada da Aplicação PDI.

Uso:
    # Janela Desktop nativa (padrão)
    python main.py

    # Modo Web (abre no navegador padrão)
    python main.py --web

    # Web com porta customizada
    python main.py --web --port 8080
"""

import argparse
import sys
from pathlib import Path

# Garante que o diretório raiz do projeto está no sys.path,
# permitindo imports como `from src.core.grayscale import ...`
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import flet as ft

from src.ui.app import APP_TITLE, build_app


def parse_args() -> argparse.Namespace:
    """
    Analisa os argumentos da linha de comando.

    Returns:
        Namespace com os atributos `web` (bool) e `port` (int).
    """
    parser = argparse.ArgumentParser(
        description=f"{APP_TITLE} — Interface Gráfica de Quantização de Imagens",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python main.py                  # Desktop nativo
  python main.py --web            # Navegador Web (porta 8550)
  python main.py --web --port 9000 # Navegador Web com porta customizada
        """,
    )
    parser.add_argument(
        "--web",
        action="store_true",
        default=False,
        help="Executa o aplicativo no navegador Web em vez de uma janela Desktop.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8550,
        help="Porta para o servidor Web (somente com --web). Padrão: 8550.",
    )
    return parser.parse_args()


def main() -> None:
    """Inicializa e executa o aplicativo Flet no modo configurado."""
    args = parse_args()
    assets_dir = str(ROOT_DIR / "assets")

    if args.web:
        print(f"Iniciando em modo Web → http://localhost:{args.port}")
        ft.run(
            main=build_app,
            view=ft.AppView.WEB_BROWSER,
            port=args.port,
            assets_dir=assets_dir,
        )
    else:
        print("Iniciando em modo Desktop...")
        ft.run(
            main=build_app,
            assets_dir=assets_dir,
        )


if __name__ == "__main__":
    main()

