import os
from typing import Dict, List, Tuple, Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Credit Risk Predictor", page_icon="💳")

st.title("💳 Credit Risk Prediction")
st.caption("Provide customer details below to estimate the credit risk.")

DISPLAY_LABELS: Dict[str, Dict[str, str]] = {
    "Sex": {"male": "Male", "female": "Female"},
    "Housing": {"free": "Living free of charge", "own": "Own home", "rent": "Rented home"},
    "Saving accounts": {
        "little": "Little savings (< €100)",
        "moderate": "Moderate savings (€100 – €500)",
        "quite rich": "Quite rich savings (€500 – €1000)",
        "rich": "Rich savings (≥ €1000)",
    },
    "Checking account": {
        "little": "Little balance (< €0)",
        "moderate": "Moderate balance (€0 – €200)",
        "rich": "Rich balance (≥ €200)",
    },
    "Job": {
        "1": "Unskilled worker (resident)",
        "2": "Skilled employee",
        "3": "Highly skilled / Management",
    },
}

EXPECTED_OPTIONS: Dict[str, List[str]] = {
    "Sex": ["male", "female"],
    "Housing": ["free", "own", "rent"],
    "Saving accounts": ["little", "moderate", "quite rich", "rich"],
    "Checking account": ["little", "moderate", "rich"],
}
MANUAL_ENCODINGS: Dict[str, Dict[str, int]] = {
    "Sex": {"female": 0, "male": 1},
    "Housing": {"free": 0, "own": 1, "rent": 2},
    "Saving accounts": {"little": 0, "moderate": 1, "quite rich": 2, "rich": 3},
    "Checking account": {"little": 0, "moderate": 1, "rich": 2},
}
JOB_OPTIONS: List[int] = [1, 2, 3]

GOOD_LABELS = {"good", "good credit", "1"}
BAD_LABELS = {"bad", "bad credit", "0"}


def normalize(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip().lower()


def format_choice(column: str, value: Any) -> str:
    label_map = DISPLAY_LABELS.get(column, {})
    key = normalize(value)
    if key in label_map:
        return label_map[key]
    return str(value).replace("_", " ").title()


@st.cache_resource
def load_resources():
    dataset_path = "german_credit_data.csv"
    model_path = "LightGBM_credit_model.pkl"
    target_encoder_path = "target_encoder.pkl"

    for path in (dataset_path, model_path, target_encoder_path):
        if not os.path.exists(path):
            st.error(f"Missing required file: {path}")
            st.stop()

    df = pd.read_csv(dataset_path, keep_default_na=False)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns="Unnamed: 0")

    for col in ("Age", "Credit amount", "Duration", "Job"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    model = joblib.load(model_path)
    target_encoder = joblib.load(target_encoder_path)

    if hasattr(target_encoder, "classes_"):
        for cls in target_encoder.classes_:
            cls_key = normalize(cls)
            if "good" in cls_key:
                GOOD_LABELS.add(cls_key)
            if "bad" in cls_key:
                BAD_LABELS.add(cls_key)

    categories: Dict[str, List[Any]] = {
        column: EXPECTED_OPTIONS[column][:] for column in EXPECTED_OPTIONS
    }
    option_labels: Dict[str, Dict[Any, str]] = {
        column: {option: format_choice(column, option) for option in options}
        for column, options in categories.items()
    }
    encoder_mappings: Dict[str, Dict[Any, Any]] = {
        column: MANUAL_ENCODINGS[column] for column in EXPECTED_OPTIONS
    }

    categories["Job"] = JOB_OPTIONS[:]
    option_labels["Job"] = {value: format_choice("Job", value) for value in JOB_OPTIONS}

    ranges = {
        "Age": (int(df["Age"].min()), int(df["Age"].max())),
        "Credit amount": (int(df["Credit amount"].min()), int(df["Credit amount"].max())),
        "Duration": (int(df["Duration"].min()), int(df["Duration"].max())),
    }

    defaults: Dict[str, Any] = {}
    for column, options in categories.items():
        default_option = options[0]
        if column in df.columns and not df[column].empty:
            mode_series = df[column].mode(dropna=True)
            if not mode_series.empty:
                mode_norm = normalize(mode_series.iloc[0])
                for option in options:
                    if normalize(option) == mode_norm:
                        default_option = option
                        break
        defaults[column] = default_option
    defaults.setdefault("Job", categories["Job"][0])

    return (
        model,
        target_encoder,
        encoder_mappings,
        categories,
        option_labels,
        ranges,
        defaults,
        df,
    )


(
    model,
    target_encoder,
    encoder_mappings,
    categories,
    option_labels,
    ranges,
    defaults,
    df,
) = load_resources()


def display_value(column: str, value: Any) -> str:
    labels = option_labels.get(column, {})
    return labels.get(value, format_choice(column, value))


def encode_value(column: str, option: Any) -> Any:
    if column == "Job":
        return int(option)
    mapping = encoder_mappings[column]
    key = normalize(option)
    if key not in mapping:
        raise ValueError(f"Value '{option}' is not available for column '{column}'.")
    return mapping[key]


def interpret_risk(pred_value: Any) -> Tuple[str, bool]:
    raw_value = pred_value
    if target_encoder and hasattr(target_encoder, "inverse_transform"):
        try:
            raw_value = target_encoder.inverse_transform([pred_value])[0]
        except Exception:
            raw_value = pred_value

    raw_str = str(raw_value).strip()
    lower = raw_str.lower()

    if lower in GOOD_LABELS:
        return raw_str, True
    if lower in BAD_LABELS:
        return raw_str, False
    if isinstance(raw_value, (int, float)):
        return raw_str, float(raw_value) >= 0.5
    return raw_str, False


def select_option(column: str, label: str, help_text: str | None = None) -> Any:
    options = categories[column]
    default_value = defaults.get(column, options[0])
    default_index = options.index(default_value) if default_value in options else 0
    return st.selectbox(
        label,
        options,
        index=default_index,
        format_func=lambda option: display_value(column, option),
        help=help_text,
    )


with st.form("prediction_form"):
    st.subheader("Customer Details")

    default_age = int(df["Age"].median())
    default_credit = int(df["Credit amount"].median())
    default_duration = int(df["Duration"].median())

    age = st.number_input(
        "Age",
        min_value=ranges["Age"][0],
        max_value=ranges["Age"][1],
        value=default_age,
        help="Applicant's age in years.",
    )

    col_left, col_right = st.columns(2)

    with col_left:
        sex = select_option("Sex", "Sex")
        job = select_option(
            "Job",
            "Job Level",
            help_text="Choose between levels 1, 2, or 3 according to applicant profile.",
        )
        housing = select_option("Housing", "Housing Status")

    with col_right:
        saving_accounts = select_option("Saving accounts", "Saving Accounts")
        checking_account = select_option("Checking account", "Checking Account")
        credit_amount = st.number_input(
            "Credit Amount (€)",
            min_value=ranges["Credit amount"][0],
            max_value=ranges["Credit amount"][1],
            value=default_credit,
            help="Requested loan amount in euros.",
        )

    duration = st.number_input(
        "Duration (months)",
        min_value=ranges["Duration"][0],
        max_value=ranges["Duration"][1],
        value=default_duration,
        help="Planned duration of the loan.",
    )

    submitted = st.form_submit_button("Predict Credit Risk", use_container_width=True)

if submitted:
    try:
        input_row = {
            "Age": int(age),
            "Sex": encode_value("Sex", sex),
            "Job": encode_value("Job", job),
            "Housing": encode_value("Housing", housing),
            "Saving accounts": encode_value("Saving accounts", saving_accounts),
            "Checking account": encode_value("Checking account", checking_account),
            "Credit amount": float(credit_amount),
            "Duration": int(duration),
        }
        input_df = pd.DataFrame([input_row])

        prediction = model.predict(input_df)[0]
        label_text, is_good = interpret_risk(prediction)

        st.subheader("Prediction")
        if is_good:
            st.success("GOOD credit risk detected.")
        else:
            st.error("BAD credit risk detected.")
        st.caption(f"Model output label: {label_text}")

        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(input_df)[0]
                classes = getattr(model, "classes_", [])
                prob_map = {normalize(cls): p for cls, p in zip(classes, proba)}
                good_prob = prob_map.get("good") or prob_map.get("1")
                bad_prob = prob_map.get("bad") or prob_map.get("0")

                if good_prob is not None and bad_prob is not None:
                    col_good, col_bad = st.columns(2)
                    col_good.metric("Good risk probability", f"{good_prob * 100:.1f}%")
                    col_bad.metric("Bad risk probability", f"{bad_prob * 100:.1f}%")
                else:
                    st.write(
                        pd.DataFrame(
                            {
                                "Class": [str(cls) for cls in classes],
                                "Probability": [f"{p * 100:.1f}%" for p in proba],
                            }
                        )
                    )
            except Exception:
                st.info("Probability details unavailable for this model configuration.")

        with st.expander("Input Summary"):
            st.table(
                pd.DataFrame(
                    {
                        "Parameter": [
                            "Age",
                            "Sex",
                            "Job Level",
                            "Housing",
                            "Saving Accounts",
                            "Checking Account",
                            "Credit Amount",
                            "Duration",
                        ],
                        "Value": [
                            f"{age} years",
                            display_value("Sex", sex),
                            display_value("Job", job),
                            display_value("Housing", housing),
                            display_value("Saving accounts", saving_accounts),
                            display_value("Checking account", checking_account),
                            f"€{credit_amount:,.0f}",
                            f"{duration} months",
                        ],
                    }
                )
            )
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
