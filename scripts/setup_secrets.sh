#!/bin/bash
# Script d'aide à la configuration des GitHub Secrets
# À exécuter UNE SEULE FOIS après avoir créé le dépôt

echo "📋 Configuration des GitHub Secrets pour le pipeline CI/CD"
echo "============================================================"
echo ""
echo "Exécute les commandes suivantes en remplaçant les valeurs :"
echo ""
echo "gh secret set HF_TOKEN          # Token Hugging Face (write access)"
echo "gh secret set ANTHROPIC_API_KEY # Clé API Anthropic pour génération email"
echo "gh secret set SMTP_SERVER       # Ex: smtp.gmail.com"
echo "gh secret set SMTP_PORT         # Ex: 587"
echo "gh secret set SMTP_USERNAME     # Ton adresse email"
echo "gh secret set SMTP_PASSWORD     # Mot de passe d'application Gmail"
echo "gh secret set MAIL_TO           # Email(s) des membres de l'équipe"
echo ""
echo "Vérification après configuration :"
echo "gh secret list"
