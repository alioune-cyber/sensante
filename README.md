---
title: SenSante
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# SenSante

Assistant de pre-diagnostic medical pour le Senegal.

## Description

SenSante utilise le Machine Learning pour aider au pre-diagnostic des maladies courantes (paludisme, grippe, typhoide) a partir des symptomes du patient.

## Structure du projet

- `data/` : Donnees patients (CSV)
- `models/` : Modele ML serialise
- `api/` : API FastAPI
- `frontend/` : Interface web
- `notebooks/` : Scripts d'exploration

## Fonctionnalites

- Prediction medicale avec Machine Learning
- Explication des resultats avec Llama 3 via Groq
- Interface web responsive
- API FastAPI

## Technologies

- FastAPI
- Scikit-learn
- Groq / Llama 3
- Tailwind CSS
- Docker

## Auteur

Moussa Diallo - L2 GLSI - ESP/UCAD

## Cours

Integration de Modeles IA - Dr. El Hadji Bassirou TOURE