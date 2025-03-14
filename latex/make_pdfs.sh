#!/usr/bin/env bash
set -e  # pour stopper le script en cas d'erreur

# Ce script compile tous les fichiers LaTeX dans les sous-répertoires chapter*
# Il doit être exécuté depuis le répertoire latex

for dir in chapter*/; do
  if [ -f "$dir/main.tex" ]; then
    echo "Compilation de $dir/main.tex"
    cd "$dir"
    latexmk -pdf main.tex
    cd ..
  else
    echo "Aucun fichier main.tex trouvé dans $dir, passage au suivant"
  fi
done

echo "Compilation des PDFs terminée"
