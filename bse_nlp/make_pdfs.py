#!/usr/bin/env python3
"""
Script pour compiler les fichiers LaTeX en PDF.
Ce script peut être exécuté via Poetry avec la commande: poetry run make-pdfs
"""

import os
import subprocess
import sys
from pathlib import Path


def main():
    """Compile tous les fichiers LaTeX en PDF."""
    # Obtenir le chemin du répertoire racine du projet
    root_dir = Path(__file__).parent.parent.absolute()
    latex_dir = root_dir / "latex"

    if not latex_dir.exists():
        print(f"Erreur: Le répertoire {latex_dir} n'existe pas.")
        sys.exit(1)

    # Vérifier si le script shell existe
    shell_script = latex_dir / "make_pdfs.sh"
    if shell_script.exists():
        print(f"Exécution du script shell {shell_script}")
        try:
            # Changer de répertoire pour être dans le répertoire latex
            os.chdir(latex_dir)
            # Rendre le script exécutable
            subprocess.run(["chmod", "+x", "make_pdfs.sh"], check=True)
            # Exécuter le script
            result = subprocess.run(["./make_pdfs.sh"], check=True)

            if result.returncode == 0:
                print("Compilation des PDFs réussie!")

                # Copier les PDFs dans le répertoire docs/pdfs
                pdfs_dir = root_dir / "docs" / "pdfs"
                pdfs_dir.mkdir(exist_ok=True)

                # Trouver tous les PDFs générés
                for chapter_dir in latex_dir.glob("chapter*"):
                    pdf_file = chapter_dir / "main.pdf"
                    if pdf_file.exists():
                        # Copier le PDF dans le répertoire docs/pdfs
                        chapter_name = chapter_dir.name
                        target_file = pdfs_dir / f"{chapter_name}.pdf"
                        print(f"Copie de {pdf_file} vers {target_file}")
                        subprocess.run(
                            ["cp", str(pdf_file), str(target_file)], check=True
                        )

                print("Tous les PDFs ont été copiés dans docs/pdfs/")
                return 0
            else:
                print("Erreur lors de la compilation des PDFs.")
                return 1
        except subprocess.CalledProcessError as e:
            print(f"Erreur lors de l'exécution du script: {e}")
            return 1
    else:
        print(f"Erreur: Le script {shell_script} n'existe pas.")

        # Compilation manuelle des fichiers LaTeX
        print("Tentative de compilation manuelle des fichiers LaTeX...")

        # Changer de répertoire pour être dans le répertoire latex
        os.chdir(latex_dir)

        success = True
        for chapter_dir in latex_dir.glob("chapter*"):
            if (chapter_dir / "main.tex").exists():
                print(f"Compilation de {chapter_dir.name}/main.tex")
                try:
                    # Changer de répertoire pour être dans le répertoire du chapitre
                    os.chdir(chapter_dir)
                    # Compiler le fichier LaTeX
                    subprocess.run(["latexmk", "-pdf", "main.tex"], check=True)
                    # Revenir au répertoire latex
                    os.chdir(latex_dir)
                except subprocess.CalledProcessError:
                    print(
                        f"Erreur lors de la compilation de {chapter_dir.name}/main.tex"
                    )
                    success = False

        if success:
            print("Compilation manuelle des PDFs réussie!")
            return 0
        else:
            print("Erreur lors de la compilation manuelle des PDFs.")
            return 1


if __name__ == "__main__":
    sys.exit(main())
