from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


EXCEL_PATH = Path(
    "/Users/magdalenacullen/Downloads/Copy of 2026 listas_entrenamiento_testeo.xlsx"
)
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)


def normalize(value):
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9ñ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def contains_any(text, alternatives):
    return any(term in text for term in alternatives)


def definition_correct(expected, response):
    """Accept a definition when its core meaning is preserved."""
    expected = normalize(expected)
    response = normalize(response)
    if not response:
        return False
    if expected == response:
        return True

    rules = {
        "charco de agua": lambda r: "charco" in r,
        "viento del noroeste": lambda r: "viento" in r and "noroeste" in r,
        "instrumento musical antiguo": lambda r: (
            "instrumento" in r and contains_any(r, ["musical", "antiguo"])
        ),
        "canto de victoria": lambda r: (
            "victoria" in r and contains_any(r, ["canto", "grito", "cancion"])
        ),
        "remedio para hemorragias": lambda r: (
            contains_any(r, ["remedio", "medicamento", "medicina", "inyeccion"])
            and "hemorrag" in r
        ),
        "arpon para pescar": lambda r: "arpon" in r and "pesc" in r,
        "armadura para la rodilla": lambda r: (
            "rodilla" in r
            and contains_any(r, ["armadura", "proteccion", "protector"])
        ),
        "aglomeracion de gente": lambda r: contains_any(
            r, ["aglomeracion", "agrupacion", "multitud", "muchedumbre"]
        )
        or ("grupo" in r and "gente" in r),
        "recibo de pago": lambda r: (
            contains_any(r, ["recibo", "ticket", "comprobante"])
            and contains_any(r, ["pago", "sueldo"])
        ),
        "baranda de una escalera": lambda r: "baranda" in r and "escalera" in r,
        "ayudante de cocina": lambda r: (
            contains_any(r, ["ayudante", "asistente"])
            and contains_any(r, ["cocina", "cocinero", "chef"])
        ),
        "poema funebre": lambda r: (
            contains_any(r, ["poema", "canto"])
            and contains_any(r, ["funebre", "funeral", "funerario", "velorio"])
        ),
        "reloj de sol": lambda r: "reloj" in r and "sol" in r,
        "habitacion de un templo": lambda r: (
            contains_any(r, ["habitacion", "cuarto"]) and "templo" in r
        ),
        "lentitud para hacer algo": lambda r: contains_any(
            r, ["lentitud", "lento", "lentamente"]
        ),
        "rama nueva de un arbol": lambda r: (
            "rama" in r and contains_any(r, ["nueva", "arbol"])
        ),
        "red para pescar": lambda r: "red" in r and contains_any(r, ["pesc", "cangrejo"]),
        "red para pescar cangrejos": lambda r: (
            "red" in r and contains_any(r, ["pesc", "cangrejo"])
        ),
        "renacuajo de una rana": lambda r: "renacuajo" in r
        or ("cria" in r and "rana" in r),
        "comedor para la servidumbre": lambda r: (
            "comedor" in r and contains_any(r, ["servidumbre", "sirviente"])
        ),
        "tipo de musica espanola": lambda r: (
            "espanol" in r and contains_any(r, ["musica", "danza", "baile"])
        ),
    }
    rule = rules.get(expected)
    return rule(response) if rule else False


def score_sheet(sheet_name, condition):
    raw = pd.read_excel(EXCEL_PATH, sheet_name=sheet_name, header=None)
    rows = []
    subject_number = 0

    for header_idx, header in raw.iterrows():
        palabra_cols = [i for i, value in enumerate(header) if normalize(value) == "palabra"]
        if len(palabra_cols) < 2:
            continue

        train_word_col = palabra_cols[0]
        test_word_col = palabra_cols[1]
        block = raw.iloc[header_idx + 1 : header_idx + 21].copy()
        if len(block) == 0:
            continue

        subject_labels = []
        for value in block.to_numpy().ravel():
            if pd.notna(value) and "sujeto" in normalize(value):
                subject_labels.append(str(value).strip())

        subject_number += 1
        subject = subject_labels[0] if subject_labels else f"{condition} {subject_number}"

        train_word_expected = block.iloc[:, train_word_col]
        train_word_response = block.iloc[:, train_word_col + 1]
        train_def_expected = block.iloc[:, train_word_col + 2]
        train_def_response = block.iloc[:, train_word_col + 3]

        test_word_expected = block.iloc[:, test_word_col]
        test_word_response = block.iloc[:, test_word_col + 1]
        test_def_expected = block.iloc[:, test_word_col + 2]
        test_def_response = block.iloc[:, test_word_col + 3]

        train_valid_word = train_word_expected.notna()
        train_valid_def = train_def_expected.notna()
        test_valid_word = test_word_expected.notna()
        test_valid_def = test_def_expected.notna()

        test_word_has_response = test_word_response[test_valid_word].map(normalize).ne("")
        test_def_has_response = test_def_response[test_valid_def].map(normalize).ne("")
        if test_word_has_response.sum() == 0 and test_def_has_response.sum() == 0:
            continue

        train_word_correct = [
            normalize(expected) == normalize(response)
            for expected, response in zip(
                train_word_expected[train_valid_word],
                train_word_response[train_valid_word],
            )
        ]
        train_def_correct = [
            definition_correct(expected, response)
            for expected, response in zip(
                train_def_expected[train_valid_def],
                train_def_response[train_valid_def],
            )
        ]

        test_word_correct = [
            normalize(expected) == normalize(response)
            for expected, response in zip(
                test_word_expected[test_valid_word],
                test_word_response[test_valid_word],
            )
        ]
        test_def_correct = [
            definition_correct(expected, response)
            for expected, response in zip(
                test_def_expected[test_valid_def],
                test_def_response[test_valid_def],
            )
        ]

        n_train_word = len(train_word_correct)
        n_train_def = len(train_def_correct)
        n_test_word = len(test_word_correct)
        n_test_def = len(test_def_correct)
        n_train_total = n_train_word + n_train_def
        n_test_total = n_test_word + n_test_def
        train_total_correct = int(np.sum(train_word_correct) + np.sum(train_def_correct))
        test_total_correct = int(np.sum(test_word_correct) + np.sum(test_def_correct))
        pct_train_total = 100 * train_total_correct / n_train_total if n_train_total else np.nan
        pct_test_total = 100 * test_total_correct / n_test_total if n_test_total else np.nan

        rows.append(
            {
                "condicion": condition,
                "hoja": sheet_name,
                "sujeto": subject,
                "n_entrenamiento_palabra": n_train_word,
                "correctas_entrenamiento_palabra": int(np.sum(train_word_correct)),
                "pct_entrenamiento_palabra": 100 * np.mean(train_word_correct)
                if n_train_word
                else np.nan,
                "n_entrenamiento_definicion": n_train_def,
                "correctas_entrenamiento_definicion": int(np.sum(train_def_correct)),
                "pct_entrenamiento_definicion": 100 * np.mean(train_def_correct)
                if n_train_def
                else np.nan,
                "n_entrenamiento_total": n_train_total,
                "correctas_entrenamiento_total": train_total_correct,
                "pct_entrenamiento_total": pct_train_total,
                "n_testeo_palabra": n_test_word,
                "correctas_testeo_palabra": int(np.sum(test_word_correct)),
                "pct_testeo_palabra": 100 * np.mean(test_word_correct) if n_test_word else np.nan,
                "n_testeo_definicion": n_test_def,
                "correctas_testeo_definicion": int(np.sum(test_def_correct)),
                "pct_testeo_definicion": 100 * np.mean(test_def_correct) if n_test_def else np.nan,
                "n_testeo_total": n_test_total,
                "correctas_testeo_total": test_total_correct,
                "pct_testeo_total": pct_test_total,
                "mejora_total": pct_test_total - pct_train_total,
                "mejora_palabra": (
                    100 * np.mean(test_word_correct) - 100 * np.mean(train_word_correct)
                    if n_test_word and n_train_word
                    else np.nan
                ),
                "mejora_definicion": (
                    100 * np.mean(test_def_correct) - 100 * np.mean(train_def_correct)
                    if n_test_def and n_train_def
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


scores = pd.concat(
    [
        score_sheet("Vigilia", "Vigilia"),
        score_sheet("Sueño", "Sueño"),
    ],
    ignore_index=True,
)

scores_path = OUT_DIR / "puntajes_sueno_vs_vigilia.csv"
scores.to_csv(scores_path, index=False)

metrics = [
    ("pct_testeo_palabra", "Palabra"),
    ("pct_testeo_definicion", "Definición"),
    ("pct_testeo_total", "Total"),
]
conditions = ["Vigilia", "Sueño"]
colors = {"Vigilia": "#6b7280", "Sueño": "#2563eb"}

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(metrics))
width = 0.34

for offset, condition in zip([-width / 2, width / 2], conditions):
    subset = scores[scores["condicion"] == condition]
    means = [subset[col].mean() for col, _ in metrics]
    sems = [subset[col].sem() for col, _ in metrics]
    ax.bar(
        x + offset,
        means,
        width,
        yerr=sems,
        capsize=5,
        label=condition,
        color=colors[condition],
        alpha=0.85,
    )
    for metric_idx, (col, _) in enumerate(metrics):
        values = subset[col].dropna().to_numpy()
        jitter = np.linspace(-0.055, 0.055, len(values)) if len(values) > 1 else np.array([0])
        ax.scatter(
            np.full(len(values), x[metric_idx] + offset) + jitter,
            values,
            color="black",
            s=32,
            alpha=0.75,
            zorder=3,
        )

ax.set_title("Comparación de desempeño en testeo: vigilia vs sueño")
ax.set_ylabel("Respuestas correctas (%)")
ax.set_xticks(x)
ax.set_xticklabels([label for _, label in metrics])
ax.set_ylim(0, 105)
ax.grid(axis="y", alpha=0.25)
ax.legend(frameon=False)
fig.tight_layout()

plot_path = OUT_DIR / "comparacion_sueno_vs_vigilia.png"
fig.savefig(plot_path, dpi=200)


def plot_bar_by_phase(phase, title, filename):
    phase_metrics = [
        (f"pct_{phase}_palabra", "Palabra"),
        (f"pct_{phase}_definicion", "Definición"),
        (f"pct_{phase}_total", "Total"),
    ]
    fig_phase, ax_phase = plt.subplots(figsize=(10, 6))
    x_phase = np.arange(len(phase_metrics))
    width_phase = 0.34

    for offset, condition in zip([-width_phase / 2, width_phase / 2], conditions):
        subset = scores[scores["condicion"] == condition]
        means = [subset[col].mean() for col, _ in phase_metrics]
        sems = [subset[col].sem() for col, _ in phase_metrics]
        ax_phase.bar(
            x_phase + offset,
            means,
            width_phase,
            yerr=sems,
            capsize=5,
            label=condition,
            color=colors[condition],
            alpha=0.85,
        )
        for metric_idx, (col, _) in enumerate(phase_metrics):
            values = subset[col].dropna().to_numpy()
            jitter = np.linspace(-0.055, 0.055, len(values)) if len(values) > 1 else np.array([0])
            ax_phase.scatter(
                np.full(len(values), x_phase[metric_idx] + offset) + jitter,
                values,
                color="black",
                s=32,
                alpha=0.75,
                zorder=3,
            )

    ax_phase.set_title(title)
    ax_phase.set_ylabel("Respuestas correctas (%)")
    ax_phase.set_xticks(x_phase)
    ax_phase.set_xticklabels([label for _, label in phase_metrics])
    ax_phase.set_ylim(0, 105)
    ax_phase.grid(axis="y", alpha=0.25)
    ax_phase.legend(frameon=False)
    fig_phase.tight_layout()
    out_path = OUT_DIR / filename
    fig_phase.savefig(out_path, dpi=200)
    return out_path

training_plot_path = plot_bar_by_phase(
    "entrenamiento",
    "Desempeño en entrenamiento: vigilia vs sueño",
    "barras_entrenamiento_vigilia_vs_sueno.png",
)
testing_plot_path = plot_bar_by_phase(
    "testeo",
    "Desempeño en testeo: vigilia vs sueño",
    "barras_testeo_vigilia_vs_sueno.png",
)


fig2, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
for ax, condition in zip(axes, conditions):
    subset = scores[scores["condicion"] == condition]
    for _, row in subset.iterrows():
        ax.plot(
            ["Entrenamiento", "Testeo"],
            [row["pct_entrenamiento_total"], row["pct_testeo_total"]],
            marker="o",
            color=colors[condition],
            alpha=0.55,
        )
    ax.scatter(
        ["Entrenamiento", "Testeo"],
        [subset["pct_entrenamiento_total"].mean(), subset["pct_testeo_total"].mean()],
        s=120,
        color="black",
        zorder=3,
        label="Promedio",
    )
    ax.set_title(condition)
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, 105)
    ax.legend(frameon=False)
axes[0].set_ylabel("Respuestas correctas (%)")
fig2.suptitle("Cambio de desempeño: entrenamiento vs testeo")
fig2.tight_layout()

paired_plot_path = OUT_DIR / "entrenamiento_vs_testeo.png"
fig2.savefig(paired_plot_path, dpi=200)

summary = scores.groupby("condicion")[
    [
        "pct_entrenamiento_total",
        "pct_testeo_total",
        "mejora_total",
        "pct_testeo_palabra",
        "pct_testeo_definicion",
    ]
].agg(["count", "mean", "sem"])
print(scores)
print("\nResumen:")
print(summary)
print(f"\nGuardado: {plot_path}")
print(f"Guardado: {paired_plot_path}")
print(f"Tabla: {scores_path}")
