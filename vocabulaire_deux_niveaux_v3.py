import csv
import json
import random
import re
import threading
import time
import unicodedata
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox

try:
    import winsound
except ImportError:
    winsound = None

# Dépendances facultatives pour la dictée vocale
try:
    import numpy as np
    import sounddevice as sd
    from faster_whisper import WhisperModel
    VOICE_AVAILABLE = True
except ImportError:
    np = None
    sd = None
    WhisperModel = None
    VOICE_AVAILABLE = False

DOSSIER = Path(__file__).resolve().parent
FICHIER_BASE = DOSSIER / "vocabulaire.csv"
FICHIER_INTERMEDIAIRE = DOSSIER / "vocabulaire_intermediaire.csv"
ASSETS = DOSSIER / "assets"
SON_POSITIF = ASSETS / "positive.wav"
SON_NEGATIF = ASSETS / "negative.wav"

FREQUENCE_AUDIO = 16000
DUREE_MAX_ENREGISTREMENT = 6.0
DUREE_SILENCE = 0.8
SEUIL_DEBUT_VOIX = 0.012
SEUIL_FIN_VOIX = 0.008
DUREE_BLOC = 0.1
TAILLE_SERIE = 20

COULEURS = {
    "fond": "#F5F2EA",
    "carte": "#FFFFFF",
    "texte": "#263238",
    "secondaire": "#68757D",
    "accent": "#367A72",
    "accent_survol": "#28635C",
    "vert": "#DDF3E5",
    "rouge": "#FBE5E3",
    "bordure": "#E1E5E4",
    "sidebar": "#EBEEE9",
    "jaune": "#FFF3CD",
}


def normaliser(texte):
    texte = str(texte or "").strip().lower()
    texte = unicodedata.normalize("NFD", texte)
    texte = "".join(c for c in texte if unicodedata.category(c) != "Mn")
    texte = re.sub(r"[^a-z0-9\s]", " ", texte)
    return " ".join(texte.split())


def separer_traductions(texte):
    return [x.strip() for x in re.split(r"[,|/]", str(texte or "")) if x.strip()]


def lire_csv(chemin):
    """Lit un CSV à colonnes français;espagnol;thème (thème facultatif)."""
    if not chemin.exists():
        return [], f"Fichier introuvable : {chemin.name}"

    erreurs = []
    try:
        with chemin.open("r", encoding="utf-8-sig", newline="") as f:
            echantillon = f.read(4096)
            f.seek(0)
            try:
                dialecte = csv.Sniffer().sniff(echantillon, delimiters=";,\t")
            except csv.Error:
                dialecte = csv.excel
                dialecte.delimiter = ";"
            lecteur = csv.DictReader(f, dialect=dialecte)
            if not lecteur.fieldnames:
                return [], "Le fichier CSV est vide ou ne contient pas d'en-tête."

            colonnes = {normaliser(c).replace(" ", ""): c for c in lecteur.fieldnames if c}
            col_fr = next((colonnes[k] for k in ("francais", "fr", "french") if k in colonnes), None)
            col_es = next((colonnes[k] for k in ("espagnol", "es", "espanol", "spanish") if k in colonnes), None)
            col_theme = next((colonnes[k] for k in ("theme", "categoria", "category") if k in colonnes), None)
            if not col_fr or not col_es:
                return [], "En-têtes requis : français;espagnol;thème (thème facultatif)."

            vocabulaire = []
            for numero, ligne in enumerate(lecteur, start=2):
                francais = str(ligne.get(col_fr, "") or "").strip()
                espagnol = str(ligne.get(col_es, "") or "").strip()
                theme = str(ligne.get(col_theme, "") or "").strip() if col_theme else "Autres"
                if not francais or not espagnol:
                    erreurs.append(f"ligne {numero} ignorée (français ou espagnol vide)")
                    continue
                vocabulaire.append({
                    "francais": francais,
                    "espagnol": separer_traductions(espagnol),
                    "theme": theme or "Autres",
                })
            if not vocabulaire:
                return [], "Aucune entrée valide trouvée dans le CSV."
            message = f"{len(vocabulaire)} mots chargés"
            if erreurs:
                message += f" · {len(erreurs)} ligne(s) ignorée(s)"
            return vocabulaire, message
    except (OSError, UnicodeError, csv.Error) as exc:
        return [], f"Impossible de lire le CSV : {exc}"


class MoteurVocal:
    """Un seul modèle Whisper partagé par les deux onglets."""
    def __init__(self, racine):
        self.racine = racine
        self.verrou = threading.Lock()
        self.modele = None
        self.chargement = False
        self.enregistrement = False
        self.onglet_enregistrement = None
        self.audio = None
        self.frequence = FREQUENCE_AUDIO

    def charger_modele(self):
        if not VOICE_AVAILABLE:
            raise RuntimeError("La dictée vocale nécessite numpy, sounddevice et faster-whisper.")
        with self.verrou:
            if self.modele is None:
                self.modele = WhisperModel("base", device="cpu", compute_type="int8")
            return self.modele

    def precharger(self):
        if not VOICE_AVAILABLE:
            return
        def travail():
            try:
                self.charger_modele()
            except Exception:
                pass
        threading.Thread(target=travail, daemon=True).start()


class OngletVocabulaire:
    def __init__(self, application, notebook, titre, chemin_csv):
        self.app = application
        self.root = application.root
        self.notebook = notebook
        self.titre = titre
        self.chemin_csv = chemin_csv
        self.vocabulaire_complet = []
        self.vocabulaire_filtre = []
        self.vocabulaire = []
        self.themes = []
        self.theme_actuel = "Tous les thèmes"
        self.index = 0
        self.numero_serie = 0
        self.score = 0
        self.questions_repondues = 0
        self.question_repondue = False
        self.premiere_tentative_faite = False
        self.reponse_initiale_correcte = False
        self.mode_revision = False
        self.source_revision = []
        # Conserve la progression normale pendant la révision des erreurs.
        self.vocabulaire_entrainement_sauve = None
        self.serie_entrainement_sauvee = 0
        self.mot_actuel = None
        self.reponses_acceptees = []
        self.en_attente_vocale = False
        self.chemin_erreurs = self.chemin_csv.with_name(f"{self.chemin_csv.stem}_erreurs.json")
        self.erreurs = self._charger_erreurs()

        self.frame = ttk.Frame(notebook, style="Main.TFrame")
        notebook.add(self.frame, text=titre)
        self._construire_interface()
        self.charger_vocabulaire()

    def _construire_interface(self):
        self.frame.columnconfigure(1, weight=1)
        self.frame.rowconfigure(0, weight=1)

        self.sidebar = tk.Frame(self.frame, bg=COULEURS["sidebar"], width=245)
        self.sidebar.grid(row=0, column=0, sticky="nsw")
        self.sidebar.grid_propagate(False)

        tk.Label(self.sidebar, text="MES THÈMES", bg=COULEURS["sidebar"], fg=COULEURS["texte"],
                 font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=18, pady=(22, 8))
        self.recherche_theme = tk.StringVar()
        recherche = ttk.Entry(self.sidebar, textvariable=self.recherche_theme)
        recherche.pack(fill="x", padx=14, pady=(0, 12), ipady=5)
        self.recherche_theme.trace_add("write", lambda *_: self.afficher_themes())

        # Menu des thèmes défilant : la liste reste utilisable même si elle est longue.
        zone_defilante = tk.Frame(self.sidebar, bg=COULEURS["sidebar"])
        zone_defilante.pack(fill="both", expand=True, padx=8, pady=(0, 10))

        self.canvas_themes = tk.Canvas(
            zone_defilante, bg=COULEURS["sidebar"], highlightthickness=0, bd=0
        )
        self.scrollbar_themes = ttk.Scrollbar(
            zone_defilante, orient="vertical", command=self.canvas_themes.yview
        )
        self.canvas_themes.configure(yscrollcommand=self.scrollbar_themes.set)

        self.zone_themes = tk.Frame(self.canvas_themes, bg=COULEURS["sidebar"])
        self._fenetre_themes = self.canvas_themes.create_window(
            (0, 0), window=self.zone_themes, anchor="nw"
        )
        self.zone_themes.bind(
            "<Configure>",
            lambda _event: self.canvas_themes.configure(
                scrollregion=self.canvas_themes.bbox("all")
            ),
        )
        self.canvas_themes.bind(
            "<Configure>",
            lambda event: self.canvas_themes.itemconfigure(
                self._fenetre_themes, width=event.width
            ),
        )

        self.canvas_themes.pack(side="left", fill="both", expand=True)
        self.scrollbar_themes.pack(side="right", fill="y")
        self.canvas_themes.bind(
            "<MouseWheel>",
            lambda event: self.canvas_themes.yview_scroll(
                int(-1 * (event.delta / 120)), "units"
            ),
        )
        self.boutons_themes = []
        self.label_info_csv = tk.Label(self.sidebar, text="", bg=COULEURS["sidebar"], fg=COULEURS["secondaire"],
                                       justify="left", wraplength=210, font=("Segoe UI", 9))
        self.label_info_csv.pack(anchor="w", padx=15, pady=(4, 8))
        self.bouton_revision = ttk.Button(
            self.sidebar, text="Réviser mes erreurs (0)", command=self.reviser_erreurs
        )
        self.bouton_revision.pack(fill="x", padx=12, pady=(0, 14))

        self.contenu = tk.Frame(self.frame, bg=COULEURS["fond"])
        self.contenu.grid(row=0, column=1, sticky="nsew")
        self.contenu.columnconfigure(0, weight=1)
        self.contenu.rowconfigure(1, weight=1)

        haut = tk.Frame(self.contenu, bg=COULEURS["fond"])
        haut.grid(row=0, column=0, sticky="ew", padx=28, pady=(20, 8))
        haut.columnconfigure(0, weight=1)
        self.label_niveau = tk.Label(haut, text=self.titre.upper(), bg=COULEURS["fond"], fg=COULEURS["accent"],
                                     font=("Segoe UI", 10, "bold"))
        self.label_niveau.grid(row=0, column=0, sticky="w")
        self.label_score = tk.Label(haut, text="Score : 0 / 0", bg=COULEURS["fond"], fg=COULEURS["secondaire"],
                                    font=("Segoe UI", 10))
        self.label_score.grid(row=0, column=1, sticky="e")

        carte = tk.Frame(self.contenu, bg=COULEURS["carte"], highlightbackground=COULEURS["bordure"],
                         highlightthickness=1)
        carte.grid(row=1, column=0, sticky="nsew", padx=28, pady=(8, 18))
        carte.columnconfigure(0, weight=1)
        carte.rowconfigure(2, weight=1)
        self.carte = carte

        self.label_progression = tk.Label(carte, text="Question 0 / 0", bg=COULEURS["carte"],
                                          fg=COULEURS["secondaire"], font=("Segoe UI", 10))
        self.label_progression.pack(anchor="w", padx=30, pady=(26, 12))
        self.barre_progression = ttk.Progressbar(carte, mode="determinate", maximum=100)
        self.barre_progression.pack(fill="x", padx=30, pady=(0, 28))

        tk.Label(carte, text="TRADUISEZ EN ESPAGNOL", bg=COULEURS["carte"], fg=COULEURS["secondaire"],
                 font=("Segoe UI", 10, "bold")).pack(pady=(0, 10))
        self.label_francais = tk.Label(carte, text="", bg=COULEURS["carte"], fg=COULEURS["texte"],
                                       font=("Segoe UI", 27, "bold"), wraplength=650, justify="center")
        self.label_francais.pack(padx=25, pady=(0, 24))

        zone_reponse = tk.Frame(carte, bg=COULEURS["carte"])
        zone_reponse.pack(fill="x", padx=45)
        self.reponse = tk.StringVar()
        self.champ_reponse = ttk.Entry(zone_reponse, textvariable=self.reponse, font=("Segoe UI", 15), justify="center")
        self.champ_reponse.pack(fill="x", ipady=8)
        self.champ_reponse.bind("<Return>", self.entree_clavier)

        commandes = tk.Frame(carte, bg=COULEURS["carte"])
        commandes.pack(pady=(14, 6))
        self.bouton_valider = ttk.Button(commandes, text="Vérifier · Entrée", command=self.evaluer_reponse)
        self.bouton_valider.grid(row=0, column=0, padx=5, ipadx=10, ipady=5)
        self.bouton_vocal = ttk.Button(commandes, text="🎙 Dicter", command=self.demarrer_enregistrement)
        self.bouton_vocal.grid(row=0, column=1, padx=5, ipadx=10, ipady=5)
        self.bouton_suivant = ttk.Button(commandes, text="Question suivante →", command=self.question_suivante, state="disabled")
        self.bouton_suivant.grid(row=0, column=2, padx=5, ipadx=10, ipady=5)

        self.label_vocal = tk.Label(carte, text="", bg=COULEURS["carte"], fg=COULEURS["secondaire"],
                                    font=("Segoe UI", 9), wraplength=650)
        self.label_vocal.pack(padx=25, pady=(4, 0))
        self.label_resultat = tk.Label(carte, text="", bg=COULEURS["carte"], fg=COULEURS["texte"],
                                       font=("Segoe UI", 12, "bold"), wraplength=680, justify="center")
        self.label_resultat.pack(padx=25, pady=(8, 0))

        bas = tk.Frame(carte, bg=COULEURS["carte"])
        bas.pack(fill="x", padx=30, pady=(16, 24))
        self.label_theme = tk.Label(bas, text="Thème : —", bg=COULEURS["carte"], fg=COULEURS["secondaire"],
                                    font=("Segoe UI", 9))
        self.label_theme.pack(side="left")
        self.bouton_recommencer_serie = ttk.Button(
            bas, text="Recommencer cette série", command=self.recommencer_serie
        )
        self.bouton_recommencer_serie.pack(side="right", padx=(8, 0))
        # Bouton de navigation également placé en bas, près du bouton de redémarrage,
        # afin qu'il reste visible à la fin d'une série, même si la rangée du haut déborde.
        self.bouton_fin_serie = ttk.Button(
            bas, text="Continuer →", command=self.question_suivante, state="disabled"
        )
        self.bouton_fin_serie.pack(side="right")

    def _charger_erreurs(self):
        try:
            with self.chemin_erreurs.open("r", encoding="utf-8") as f:
                contenu = json.load(f)
            return contenu if isinstance(contenu, list) else []
        except (OSError, json.JSONDecodeError):
            return []

    def _sauver_erreurs(self):
        try:
            with self.chemin_erreurs.open("w", encoding="utf-8") as f:
                json.dump(self.erreurs, f, ensure_ascii=False, indent=2)
        except OSError as exc:
            messagebox.showwarning(
                "Sauvegarde impossible",
                f"Les erreurs n'ont pas pu être enregistrées : {exc}"
            )
        self.bouton_revision.config(text=f"Réviser mes erreurs ({len(self.erreurs)})")

    def _ajouter_erreur(self, mot):
        cle = normaliser(mot["francais"])
        if not any(normaliser(e.get("francais", "")) == cle for e in self.erreurs):
            self.erreurs.append({
                "francais": mot["francais"],
                "espagnol": list(mot["espagnol"]),
                "theme": mot.get("theme", "Autres")
            })
            self._sauver_erreurs()

    def _retirer_erreur(self, mot):
        cle = normaliser(mot["francais"])
        nouvelle_liste = [e for e in self.erreurs if normaliser(e.get("francais", "")) != cle]
        if len(nouvelle_liste) != len(self.erreurs):
            self.erreurs = nouvelle_liste
            self._sauver_erreurs()

    def charger_vocabulaire(self):
        self.vocabulaire_complet, message = lire_csv(self.chemin_csv)
        self.erreurs = self._charger_erreurs()
        self.bouton_revision.config(text=f"Réviser mes erreurs ({len(self.erreurs)})")
        self.label_info_csv.config(text=message)
        self.themes = sorted({m["theme"] for m in self.vocabulaire_complet}, key=str.casefold)
        if not self.vocabulaire_complet:
            self.vocabulaire = []
            self.vocabulaire_filtre = []
            self.mot_actuel = None
            self.afficher_themes()
            self.afficher_etat_vide(message)
            return
        if self.theme_actuel != "Tous les thèmes" and self.theme_actuel not in self.themes:
            self.theme_actuel = "Tous les thèmes"
        self.appliquer_filtre(reinitialiser=True)
        self.afficher_themes()

    def afficher_themes(self):
        for widget in self.zone_themes.winfo_children():
            widget.destroy()
        filtre = normaliser(self.recherche_theme.get())
        options = ["Tous les thèmes"] + self.themes
        for theme in options:
            if filtre and filtre not in normaliser(theme):
                continue
            actif = theme == self.theme_actuel
            bouton = tk.Button(self.zone_themes, text=theme, anchor="w", relief="flat", bd=0,
                               bg=COULEURS["accent"] if actif else COULEURS["sidebar"],
                               fg="white" if actif else COULEURS["texte"],
                               activebackground=COULEURS["accent_survol"], activeforeground="white",
                               font=("Segoe UI", 10, "bold" if actif else "normal"), cursor="hand2",
                               padx=12, pady=9, command=lambda t=theme: self.selectionner_theme(t))
            bouton.pack(fill="x", padx=4, pady=2)

    def selectionner_theme(self, theme):
        self.theme_actuel = theme
        self.appliquer_filtre(reinitialiser=True)
        self.afficher_themes()

    def appliquer_filtre(self, reinitialiser=False):
        if self.theme_actuel == "Tous les thèmes":
            source = list(self.vocabulaire_complet)
        else:
            source = [m for m in self.vocabulaire_complet if m["theme"] == self.theme_actuel]
        random.shuffle(source)
        self.vocabulaire_filtre = source
        self.mode_revision = False
        self.source_revision = []
        self.numero_serie = 0
        if reinitialiser:
            self._charger_serie(0)

    def _charger_serie(self, numero):
        debut = numero * TAILLE_SERIE
        self.vocabulaire = self.vocabulaire_filtre[debut:debut + TAILLE_SERIE]
        self.numero_serie = numero
        self.index = 0
        self.score = 0
        self.questions_repondues = 0
        self.question_repondue = False
        self.premiere_tentative_faite = False
        self.reponse_initiale_correcte = False
        self.afficher_question()

    def reviser_erreurs(self):
        self.erreurs = self._charger_erreurs()
        if not self.erreurs:
            messagebox.showinfo("Révision des erreurs", "Aucune erreur enregistrée pour ce niveau. Continuez à pratiquer !")
            self.bouton_revision.config(text="Réviser mes erreurs (0)")
            return
        # Mémorise le vocabulaire filtré et la série normale pour y revenir après la révision.
        if not self.mode_revision:
            self.vocabulaire_entrainement_sauve = list(self.vocabulaire_filtre)
            self.serie_entrainement_sauvee = self.numero_serie
        self.mode_revision = True
        self.source_revision = list(self.erreurs)
        random.shuffle(self.source_revision)
        self.vocabulaire_filtre = list(self.source_revision)
        self._charger_serie(0)
        self.label_vocal.config(text="Mode révision : seuls les mots précédemment ratés sont proposés.")

    def recommencer_serie(self):
        if self.mode_revision:
            self.vocabulaire_filtre = list(self.source_revision)
        self._charger_serie(self.numero_serie)

    def afficher_etat_vide(self, message):
        self.label_francais.config(text="Ajoutez votre vocabulaire pour commencer")
        self.label_progression.config(text="Aucune question disponible")
        self.barre_progression["value"] = 0
        self.label_score.config(text="Score : 0 / 0")
        self.label_theme.config(text="Thème : —")
        self.label_resultat.config(text=message, fg=COULEURS["secondaire"])
        self.champ_reponse.config(state="disabled")
        self.bouton_valider.config(state="disabled")
        self.bouton_suivant.config(state="disabled")
        self.bouton_vocal.config(state="disabled")

    def afficher_question(self):
        if not self.vocabulaire:
            self.afficher_etat_vide("Aucun mot dans ce thème.")
            return
        if self.index >= len(self.vocabulaire):
            total_source = len(self.vocabulaire_filtre)
            prochaine_serie_disponible = (self.numero_serie + 1) * TAILLE_SERIE < total_source
            self.label_francais.config(text="¡Muy bien!", fg=COULEURS["accent"])
            self.label_progression.config(text=f"Série {self.numero_serie + 1} terminée")
            self.barre_progression["value"] = 100
            mode = "Révision" if self.mode_revision else "Entraînement"
            self.label_resultat.config(
                text=f"{mode} terminé ! Score de première réponse : {self.score} / {self.questions_repondues}",
                fg=COULEURS["accent"]
            )
            self.champ_reponse.config(state="disabled")
            self.bouton_valider.config(state="disabled")
            self.bouton_vocal.config(state="disabled")
            if prochaine_serie_disponible:
                texte_fin = "Série suivante →"
                self.bouton_suivant.config(text=texte_fin, state="normal")
                self.bouton_fin_serie.config(text=texte_fin, state="normal")
            elif self.mode_revision:
                # Même s'il n'y a plus de série d'erreurs, l'utilisateur peut reprendre
                # l'entraînement normal au lieu d'être obligé de recommencer la révision.
                texte_fin = "Reprendre l’entraînement →"
                self.bouton_suivant.config(text=texte_fin, state="normal")
                self.bouton_fin_serie.config(text=texte_fin, state="normal")
            else:
                self.bouton_suivant.config(text="Toutes les séries terminées", state="disabled")
                self.bouton_fin_serie.config(text="Toutes les séries terminées", state="disabled")
            return
        self.mot_actuel = self.vocabulaire[self.index]
        self.reponses_acceptees = self.mot_actuel["espagnol"]
        self.question_repondue = False
        self.premiere_tentative_faite = False
        self.reponse_initiale_correcte = False
        self.label_francais.config(text=self.mot_actuel["francais"], fg=COULEURS["texte"])
        total_source = len(self.vocabulaire_filtre)
        self.label_progression.config(text=f"Série {self.numero_serie + 1} · Question {self.index + 1} / {len(self.vocabulaire)}")
        self.barre_progression["value"] = (self.index / len(self.vocabulaire)) * 100
        self.label_score.config(text=f"Score initial : {self.score} / {self.questions_repondues}")
        self.label_theme.config(text=("Révision des erreurs" if self.mode_revision else f"Thème : {self.mot_actuel['theme']}"))
        self.reponse.set("")
        self.label_resultat.config(text="")
        self.label_vocal.config(text="")
        self.champ_reponse.config(state="normal")
        self.bouton_valider.config(state="normal", text="Vérifier · Entrée")
        self.bouton_suivant.config(state="disabled", text="Question suivante →")
        self.bouton_fin_serie.config(state="disabled", text="Continuer →")
        self.bouton_vocal.config(state="normal" if VOICE_AVAILABLE else "disabled")
        self.champ_reponse.focus_set()

    def entree_clavier(self, _event=None):
        if self.question_repondue:
            self.question_suivante()
        else:
            self.evaluer_reponse()
        return "break"

    def evaluer_reponse(self):
        if not self.mot_actuel or self.question_repondue:
            return
        saisie = self.reponse.get().strip()
        if not saisie:
            self.label_resultat.config(text="Écrivez une réponse avant de vérifier.", fg="#9B5A00")
            return

        attendues = self.reponses_acceptees
        ok = normaliser(saisie) in {normaliser(x) for x in attendues}

        # La note est calculée uniquement sur la toute première réponse.
        if not self.premiere_tentative_faite:
            self.premiere_tentative_faite = True
            self.questions_repondues += 1
            self.reponse_initiale_correcte = ok
            if ok:
                self.score += 1
            else:
                self._ajouter_erreur(self.mot_actuel)

        if ok:
            if self.mode_revision and self.reponse_initiale_correcte:
                self._retirer_erreur(self.mot_actuel)
            self.question_repondue = True
            if self.reponse_initiale_correcte:
                self.label_resultat.config(text="¡Correcto! ✓", fg=COULEURS["accent"])
                self.jouer_son(SON_POSITIF)
            else:
                self.label_resultat.config(
                    text="Bonne correction ! La réponse était déjà comptée comme incorrecte dans le score initial.",
                    fg=COULEURS["accent"]
                )
                self.jouer_son(SON_POSITIF)
            self.champ_reponse.config(state="disabled")
            self.bouton_valider.config(state="disabled")
            self.bouton_suivant.config(state="normal", text="Question suivante →")
        else:
            attendues_txt = " / ".join(attendues)
            if self.premiere_tentative_faite and not self.reponse_initiale_correcte:
                self.label_resultat.config(
                    text=f"Pas tout à fait. Réponse attendue : {attendues_txt}. Retapez une réponse correcte pour continuer.",
                    fg="#B6423A"
                )
            else:
                self.label_resultat.config(text=f"Pas tout à fait. Réponse attendue : {attendues_txt}. Réessayez pour continuer.", fg="#B6423A")
            self.jouer_son(SON_NEGATIF)
            self.reponse.set("")
            self.bouton_valider.config(text="Valider la correction")
            self.champ_reponse.focus_set()
        self.label_score.config(text=f"Score initial : {self.score} / {self.questions_repondues}")

    def question_suivante(self):
        if self.index >= len(self.vocabulaire):
            if (self.numero_serie + 1) * TAILLE_SERIE < len(self.vocabulaire_filtre):
                self._charger_serie(self.numero_serie + 1)
            elif self.mode_revision:
                # Fin de la dernière série d'erreurs : retour à la série normale suivante.
                self.reprendre_entrainement()
            return
        if not self.question_repondue:
            return
        self.index += 1
        self.afficher_question()

    # def reprendre_entrainement(self):
    #     """Quitte la révision et reprend le vocabulaire normal à la série suivante."""
    #     if self.vocabulaire_entrainement_sauve is None:
    #         # Sécurité si la révision a été lancée avant qu'une progression soit mémorisée.
    #         self.mode_revision = False
    #         self.charger_vocabulaire()
    #         return

    #     vocabulaire_normal = self.vocabulaire_entrainement_sauve
    #     serie_suivante = self.serie_entrainement_sauvee + 1
    #     self.mode_revision = False
    #     self.source_revision = []
    #     self.vocabulaire_filtre = list(vocabulaire_normal)
    #     self.vocabulaire_entrainement_sauve = None
    #     if serie_suivante * TAILLE_SERIE < len(self.vocabulaire_filtre):
    #         self._charger_serie(serie_suivante)
    #     else:
    #         # Si la série normale actuelle était la dernière, on la réaffiche.
    #         self._charger_serie(self.serie_entrainement_sauvee)
    #     self.label_vocal.config(text="Retour à l’entraînement normal.")

    def reprendre_entrainement(self):
        """Quitte la révision et reprend l'entraînement normal."""
        if self.vocabulaire_entrainement_sauve is None:
            self.mode_revision = False
            self.source_revision = []
            self.charger_vocabulaire()
            return

        vocabulaire_normal = list(self.vocabulaire_entrainement_sauve)
        serie_sauvee = self.serie_entrainement_sauvee

        self.mode_revision = False
        self.source_revision = []
        self.vocabulaire_filtre = vocabulaire_normal

        # Libère la sauvegarde pour éviter de réutiliser un ancien état.
        self.vocabulaire_entrainement_sauve = None
        self.serie_entrainement_sauvee = None

        serie_suivante = serie_sauvee + 1

        if serie_suivante * TAILLE_SERIE < len(self.vocabulaire_filtre):
            numero_serie = serie_suivante
        else:
            # La série sauvegardée était la dernière : on la reprend.
            numero_serie = serie_sauvee

        self._charger_serie(numero_serie)
        self.label_vocal.config(text="Retour à l’entraînement normal.")

    def recommencer(self):
        # Recharge le fichier CSV puis recommence le niveau.
        self.charger_vocabulaire()

    def jouer_son(self, chemin):
        try:
            if winsound and chemin.exists():
                winsound.PlaySound(str(chemin), winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception:
            pass

    def demarrer_enregistrement(self):
        if not VOICE_AVAILABLE:
            messagebox.showwarning("Dictée vocale indisponible", "Installez les dépendances : numpy, sounddevice et faster-whisper.")
            return
        if not self.mot_actuel or self.question_repondue or self.app.moteur.enregistrement:
            return
        self.app.moteur.enregistrement = True
        self.app.moteur.onglet_enregistrement = self
        self.en_attente_vocale = True
        self.label_vocal.config(text="Écoute en cours… parlez maintenant (maximum 6 secondes).")
        self.bouton_vocal.config(state="disabled")
        self.bouton_valider.config(state="disabled")
        self.bouton_suivant.config(state="disabled")
        self.champ_reponse.config(state="disabled")
        self.app.verrouiller_onglets(True)
        threading.Thread(target=self._enregistrer_et_transcrire, daemon=True).start()

    def _enregistrer_et_transcrire(self):
        erreur = None
        texte = ""
        try:
            modele = self.app.moteur.charger_modele()

            # Enregistrement adaptatif : on s'arrête après 0,8 s de silence
            # suivant la parole, au lieu d'attendre systématiquement 6 secondes.
            blocs = []
            taille_bloc = int(self.app.moteur.frequence * DUREE_BLOC)
            debut = time.monotonic()
            derniere_voix = None
            parole_detectee = False
            self.root.after(0, lambda: self.label_vocal.config(
                text="Écoute… parlez, l’enregistrement s’arrêtera après votre phrase."
            ))

            with sd.InputStream(samplerate=self.app.moteur.frequence,
                                channels=1, dtype="float32", blocksize=taille_bloc) as flux:
                while time.monotonic() - debut < DUREE_MAX_ENREGISTREMENT:
                    bloc, _ = flux.read(taille_bloc)
                    bloc = np.squeeze(bloc).copy()
                    blocs.append(bloc)
                    niveau = float(np.sqrt(np.mean(np.square(bloc))))
                    maintenant = time.monotonic()

                    if niveau >= SEUIL_DEBUT_VOIX:
                        parole_detectee = True
                        derniere_voix = maintenant
                    elif parole_detectee and niveau >= SEUIL_FIN_VOIX:
                        derniere_voix = maintenant

                    if parole_detectee and derniere_voix is not None and maintenant - derniere_voix >= DUREE_SILENCE:
                        break

            if not parole_detectee:
                texte = ""
            else:
                audio = np.concatenate(blocs).astype("float32", copy=False)
                segments, _ = modele.transcribe(
                    audio, language="es", vad_filter=True,
                    beam_size=1, best_of=1, condition_on_previous_text=False
                )
                texte = " ".join(segment.text.strip() for segment in segments).strip()
        except Exception as exc:
            erreur = str(exc)
        self.root.after(0, lambda: self._fin_enregistrement(texte, erreur))

    def _fin_enregistrement(self, texte, erreur):
        self.app.moteur.enregistrement = False
        self.app.moteur.onglet_enregistrement = None
        self.app.verrouiller_onglets(False)
        self.en_attente_vocale = False
        if erreur:
            self.label_vocal.config(text=f"Erreur de dictée : {erreur}")
        elif texte:
            self.reponse.set(texte)
            self.label_vocal.config(text=f"Reconnu : « {texte} »")
        else:
            self.label_vocal.config(text="Aucune parole reconnue. Réessayez ou tapez la réponse.")
        if not self.question_repondue and self.mot_actuel:
            self.champ_reponse.config(state="normal")
            self.bouton_valider.config(state="normal")
            self.bouton_vocal.config(state="normal" if VOICE_AVAILABLE else "disabled")
            self.champ_reponse.focus_set()


class ApplicationVocabulaire:
    def __init__(self, root):
        self.root = root
        self.root.title("Mon vocabulaire espagnol")
        self.root.geometry("1120x720")
        self.root.minsize(900, 600)
        self.root.configure(bg=COULEURS["fond"])
        self.moteur = MoteurVocal(root)
        self._styles()

        entete = tk.Frame(root, bg=COULEURS["fond"])
        entete.pack(fill="x", padx=24, pady=(16, 4))
        tk.Label(entete, text="Mon vocabulaire espagnol", bg=COULEURS["fond"], fg=COULEURS["texte"],
                 font=("Segoe UI", 20, "bold")).pack(side="left")
        tk.Label(entete, text="Apprenez à votre rythme", bg=COULEURS["fond"], fg=COULEURS["secondaire"],
                 font=("Segoe UI", 10)).pack(side="right", pady=(8, 0))

        # Barre d'onglets personnalisée : les deux boutons occupent exactement
        # la moitié de la largeur disponible, sans espace vide à droite.
        barre_onglets = tk.Frame(root, bg=COULEURS["fond"])
        barre_onglets.pack(fill="x", padx=18, pady=(6, 0))
        barre_onglets.columnconfigure(0, weight=1, uniform="onglets")
        barre_onglets.columnconfigure(1, weight=1, uniform="onglets")
        self.boutons_onglets = []
        for i, texte in enumerate(("Vocabulaire de base", "Vocabulaire intermédiaire")):
            bouton = tk.Button(
                barre_onglets, text=texte, relief="flat", bd=0,
                font=("Segoe UI", 11, "bold"), cursor="hand2", pady=12,
                command=lambda index=i: self.selectionner_onglet(index)
            )
            bouton.grid(row=0, column=i, sticky="nsew", padx=(0, 0), pady=0)
            self.boutons_onglets.append(bouton)

        # Le Notebook conserve les frames et leur état, mais ses onglets natifs
        # sont masqués : seule la barre personnalisée ci-dessus est visible.
        style = ttk.Style()
        try:
            style.layout("Hidden.TNotebook.Tab", [])
            style.configure("Hidden.TNotebook", tabmargins=0, borderwidth=0)
        except tk.TclError:
            pass
        self.notebook = ttk.Notebook(root, style="Hidden.TNotebook")
        self.notebook.pack(fill="both", expand=True, padx=18, pady=(0, 18))
        self.onglets = []
        self.onglets.append(OngletVocabulaire(self, self.notebook, "Vocabulaire de base", FICHIER_BASE))
        self.onglets.append(OngletVocabulaire(self, self.notebook, "Vocabulaire intermédiaire", FICHIER_INTERMEDIAIRE))
        self.onglet_precedent = self.notebook.select()
        self.notebook.bind("<<NotebookTabChanged>>", self._changement_onglet)
        self._onglets_verrouilles = False
        self.selectionner_onglet(0)
        self.moteur.precharger()

        if not VOICE_AVAILABLE:
            self.root.after(900, lambda: self._info_voix.config(text="Dictée vocale désactivée : installez numpy, sounddevice et faster-whisper.")) if hasattr(self, "_info_voix") else None

    def _styles(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Main.TFrame", background=COULEURS["fond"])
        style.configure("TNotebook", background=COULEURS["fond"], borderwidth=0)
        style.configure("TNotebook.Tab", padding=(0, 0), font=("Segoe UI", 10, "bold"), background=COULEURS["fond"])
        style.configure("TEntry", padding=7, fieldbackground="#FFFFFF")
        style.configure("TButton", font=("Segoe UI", 9, "bold"), padding=(9, 5))
        style.map("TButton", foreground=[("disabled", "#9CA5A8")])
        style.configure("Horizontal.TProgressbar", troughcolor="#E8ECE9", background=COULEURS["accent"], thickness=8)

    def selectionner_onglet(self, index):
        if getattr(self, "_onglets_verrouilles", False):
            return
        self.notebook.select(self.onglets[index].frame)
        for i, bouton in enumerate(self.boutons_onglets):
            actif = i == index
            bouton.configure(
                bg=COULEURS["accent"] if actif else "#E6E9E5",
                fg="white" if actif else COULEURS["texte"],
                activebackground=COULEURS["accent_survol"] if actif else "#DDE2DD",
                activeforeground="white" if actif else COULEURS["texte"],
                highlightthickness=0
            )

    def verrouiller_onglets(self, verrouiller):
        # Le notebook ne reçoit pas de clic clavier/souris pendant une dictée : aucun résultat
        # ne peut être appliqué par erreur à l'autre niveau.
        self._onglets_verrouilles = verrouiller

    def _changement_onglet(self, _event=None):
        if getattr(self, "_onglets_verrouilles", False):
            cible = self.moteur.onglet_enregistrement
            if cible is not None:
                self.notebook.select(cible.frame)
            return
        self.onglet_precedent = self.notebook.select()


def main():
    root = tk.Tk()
    ApplicationVocabulaire(root)
    root.mainloop()


if __name__ == "__main__":
    main()
