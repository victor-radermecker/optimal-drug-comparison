import pandas as pd
import plotly.express as px
import streamlit as st
import re


def clean_drug_name(name):
    """Remove text in parentheses, 'ER', 'HCl', 'HFA', and '-CD/UC/HS Starter' from drug names."""
    # First remove text in parentheses
    name = re.sub(r"\s*\([^)]*\)", "", name)
    # Remove 'ER', 'HCl', and 'HFA' (with or without spaces)
    name = re.sub(r"\s*ER\b|\s*HCl\b|\s*HFA\b", "", name)
    # Remove '-CD/UC/HS Starter' text
    name = re.sub(r"\s*-CD/UC/HS Starter", "", name)
    # Clean up any extra whitespace and return
    return name.strip()


# --------------------
# Page Config
# --------------------
st.set_page_config(
    layout="wide", page_title="Essential & Optimal Formulary Disruption Dashboard"
)


# --------------------
# Load Formulary Data with Caching
# --------------------
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_formularies():
    """Load and process formulary data from Excel with caching."""
    formulary_excel = "./data/FinalFormularies.xlsx"
    formularies_raw = {
        "Base": pd.read_excel(
            formulary_excel, sheet_name="Base Formulary", dtype={"GPI": str}
        ),
        "Premier": pd.read_excel(
            formulary_excel, sheet_name="Premier Formulary", dtype={"GPI": str}
        ),
        "Premier Plus": pd.read_excel(
            formulary_excel, sheet_name="PremierPlus Formulary", dtype={"GPI": str}
        ),
        "Preferred": pd.read_excel(
            formulary_excel, sheet_name="Preferred Formulary", dtype={"GPI": str}
        ),
        "Optimal": pd.read_excel(
            formulary_excel, sheet_name="Optimal Formulary", dtype={"GPI": str}
        ),
        # "OptimalPAP": pd.read_excel(
        #     formulary_excel, sheet_name="Optimal PAP Coverage", dtype={"GPI": str}
        # ),
    }

    # Process formulary data
    processed_formularies = {}
    for name, df in formularies_raw.items():
        df["GPI"] = df["GPI"].astype(str)
        if name == "OptimalPAP":
            # Skip adding OptimalPAP as its own formulary
            continue
        elif name == "Base":
            processed_formularies[name] = df
        elif name == "Premier":
            # Combine Base and Premier
            processed_formularies[name] = pd.concat(
                [processed_formularies["Base"], df]
            ).drop_duplicates()
        elif name == "Premier Plus":
            # Combine Base, Premier, and Premier Plus
            processed_formularies[name] = pd.concat(
                [processed_formularies["Base"], processed_formularies["Premier"], df]
            ).drop_duplicates()
        elif name == "Preferred":
            # Combine Base and Preferred
            processed_formularies[name] = pd.concat(
                [processed_formularies["Base"], df]
            ).drop_duplicates()
        elif name == "Optimal":
            # Combine Base, Premier, Premier Plus, Preferred, Optimal, and OptimalPAP
            optimal_combined = pd.concat(
                [
                    processed_formularies["Base"],
                    processed_formularies["Premier"],
                    processed_formularies["Premier Plus"],
                    processed_formularies["Preferred"],
                    df,
                    # formularies_raw["OptimalPAP"],  # Include OptimalPAP
                ]
            ).drop_duplicates()
            processed_formularies[name] = optimal_combined

    return processed_formularies


# Cache Medispan data load
@st.cache_data(ttl=86400)  # Cache for 24 hours
def load_medispan():
    """Load medispan data with caching."""
    # Only load the columns we actually need to save memory
    return pd.read_parquet(
        "./data/medispan.parquet",
        columns=["ndc", "gpi", "drugname", "therapeuticclass02"],
    )


# Process uploaded file once
@st.cache_data
def process_uploaded_file(uploaded_file):
    """Process uploaded claims file with caching."""
    file_extension = uploaded_file.name.split(".")[-1].lower()

    if file_extension in ["xlsx", "xls"]:
        # Handle Excel files
        claims = pd.read_excel(uploaded_file, dtype={"GPI": str, "NDC": str})
    else:
        # Handle CSV files
        claims = pd.read_csv(uploaded_file, dtype={"GPI": str, "NDC": str})

    # Validate required columns
    if "NDC" not in claims.columns:
        return None

    return claims


# Process and merge claims with Medispan once
@st.cache_data
def prepare_claims_data(claims_df, medispan_df):
    """Merge claims with medispan and prepare data for analysis."""
    # Filter and prepare claims
    df = claims_df[claims_df["NDC"].notna()].copy()
    df = df.rename(columns={"NDC": "ndc"})
    df["ndc"] = df["ndc"].apply(lambda x: x.zfill(11))

    # Merge with medispan data
    merged_df = df.merge(medispan_df, on="ndc", how="left")

    # Rename and clean columns
    merged_df = merged_df.rename(columns={"gpi": "GPI"})
    merged_df["GPI"] = merged_df["GPI"].astype(str)

    # Clean drugname and create a cleaned_drugname column for reuse
    merged_df["drugname"] = merged_df["drugname"].astype(str)
    merged_df["cleaned_drugname"] = merged_df["drugname"].apply(
        lambda x: clean_drug_name(x).lower()
    )

    # Create claimstatus if missing
    if "claimstatus" not in merged_df.columns:
        if "repriced_ingredient_cost" in merged_df.columns:
            # Set to 1 if cost is positive, 0 if cost is null or non-positive
            merged_df["claimstatus"] = merged_df["repriced_ingredient_cost"].apply(
                lambda x: 1 if x is not None and x > 0 else 0
            )
        else:
            # If both columns are missing, create a default claimstatus of 0
            merged_df["claimstatus"] = 0

    return merged_df


# Load formularies just once
formularies = load_formularies()

# --------------------
# Biosimilar Mappings with Set Optimization
# --------------------
humira_biosimilars = [
    "Amjevita",
    "adalimumab-atto",
    "Cyltezo",
    "adalimumab-adbm",
    "Hyrimoz",
    "adalimumab-adaz",
    "Hadlima",
    "adalimumab-bwwd",
    "Abrilada",
    "adalimumab-afzb",
    "Hulio",
    "adalimumab-fkjp",
    "Yusimry",
    "adalimumab-aqvh",
    "Idacio",
    "adalimumab-aacf",
    "Yuflyma",
    "adalimumab-aaty",
    "Simlandi",
    "adalimumab-ryvk",
    "humira",
]

# Create lowercase set once for O(1) lookups
humira_biosimilars_set = {b.lower() for b in humira_biosimilars}


def is_biosimilar_covered_optimized(drug_name_clean, formulary_drugs_set):
    """
    Optimized version using sets for O(1) lookups.
    Checks if a drug (Humira or any of its biosimilars) is covered by a formulary.

    Parameters:
    - drug_name_clean: Pre-cleaned lowercase name of the drug to check
    - formulary_drugs_set: Set of preprocessed lowercase formulary drug names
    """
    # Check if the claim's drug is a biosimilar (using any segment match)
    is_biosimilar = any(biosim in drug_name_clean for biosim in humira_biosimilars_set)

    if not is_biosimilar:
        return False

    # If it is a biosimilar, check if any biosimilar is in the formulary
    # This is a simple check since we've already preprocessed the formulary drugs
    return any(
        any(biosim in fdrug for biosim in humira_biosimilars_set)
        for fdrug in formulary_drugs_set
    )


# --------------------
# Title & Upload
# --------------------
col1, col2 = st.columns([3, 1])
with col1:
    st.title("💊 Essential & Optimal Formulary Disruption Dashboard")
with col2:
    st.download_button(
        label="📥 Download Formulary File",
        data=open("./data/FinalFormularies.xlsx", "rb"),
        file_name="FinalFormularies.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# Add information about required columns
st.markdown(
    """
### Required Columns in Upload File:
- **NDC**: National Drug Code (Required)
- **member_id**: Member identifier (Optional, enables member-based analytics)
- **repriced_ingredient_cost**: Cost information (Optional, enables cost-based analytics)
- **member_paid**: Member payment amount (Optional, enables member payment analytics)
- **claimstatus**: Claim status (Optional, enables claim counting)
"""
)

# Update uploader to support both CSV and Excel files
uploaded_file = st.file_uploader(
    "📁 Upload your claims file (CSV or Excel)", type=["csv", "xlsx", "xls"]
)

if not uploaded_file:
    st.info("👈 Please upload a claims file to begin.")
    st.stop()

# Process uploaded file
raw_claims = process_uploaded_file(uploaded_file)
if raw_claims is None:
    st.error("❌ Uploaded file must contain column: NDC")
    st.stop()

# Load medispan data
medispan = load_medispan()

# Merge claims with medispan once
claims = prepare_claims_data(raw_claims, medispan)

# Calculate metrics only once
total_claims_count = len(claims)
active_claims_count = claims["claimstatus"].sum()


# --------------------
# Similarity Logic
# --------------------
def prepare_gpi_prefix_sets(formulary_gpis, prefix_lengths=(14, 12, 10, 8, 6)):
    """
    Build a dict of sets for each prefix length.
    E.g., prefix_sets[14] = set of all 14-digit GPI prefixes in the formulary
    """
    prefix_sets = {}
    for length in prefix_lengths:
        prefix_sets[length] = {
            g[:length] for g in formulary_gpis if g and isinstance(g, str)
        }
    return prefix_sets


def compute_gpi_similarity_fast(gpi_claim, prefix_sets):
    """
    Checks membership in pre-built prefix sets.
    Returns the first (largest) length for which there's a match, else 0.
    """
    if not isinstance(gpi_claim, str):
        return 0

    for length in [14, 12, 10, 8, 6]:
        if gpi_claim[:length] in prefix_sets[length]:
            return length
    return 0


# The original function can remain as a fallback
def compute_gpi_similarity(gpi_claim, gpi_list):
    """Return the highest matching digit length between a claim's GPI and any GPI in the formulary."""
    for length in [14, 12, 10, 8, 6]:
        if any(gpi_claim[:length] == g[:length] for g in gpi_list):
            return length
    return 0


similarity_labels = {
    14: "Exact match (Same product)",
    12: "Same ingredient, route, and form",
    10: "Same ingredient, different strength/form",
    8: "Same subclass",
    6: "Same class",
    -1: "NDC Description match only",
    0: "No close match",
    99: "Covered (Exact GPI Match)",  # New category for covered claims
}


# --------------------
# Analyze All Formularies with Optimized Memory Usage
# --------------------
@st.cache_data
def analyze_all_formularies(claims_df, formularies_dict):
    """
    Pre-compute coverage analysis for all formularies at once to avoid repeated copies
    of the claims DataFrame.
    """
    all_summaries = []
    results = {}

    # Store GPI_Match and Name_Match for each formulary to avoid recomputing
    coverage_info = {}

    for f_name, f_df in formularies_dict.items():
        # Compute coverage information once per formulary
        gpi_matches = claims_df["GPI"].isin(f_df["GPI"])

        # Pre-process formulary drug names only once for each formulary
        formulary_clean_names = [
            clean_drug_name(desc).lower()
            for desc in f_df["ndcdescription"]
            if isinstance(desc, str)
        ]
        formulary_names_set = set(formulary_clean_names)

        # Pre-check biosimilar coverage
        has_biosimilar_coverage = any(
            any(biosim in fdrug for biosim in humira_biosimilars_set)
            for fdrug in formulary_names_set
        )

        # Process name matches for all claims - use cleaned_drugname instead
        def check_name_match(drug_name_clean):
            # First check normal substring match - using already cleaned name
            if any(drug_name_clean in fname for fname in formulary_names_set):
                return True
            # Then check biosimilar status
            is_biosimilar = any(
                biosim in drug_name_clean for biosim in humira_biosimilars_set
            )
            if is_biosimilar:
                return has_biosimilar_coverage
            return False

        # Use the pre-cleaned column here
        name_matches = claims_df["cleaned_drugname"].apply(check_name_match)

        # Store pre-computed matches for this formulary
        coverage_info[f_name] = {
            "gpi_matches": gpi_matches,
            "name_matches": name_matches,
            "prefix_sets": prepare_gpi_prefix_sets(f_df["GPI"].tolist()),
        }

        # Do the analysis with a fresh copy (needed only once per formulary)
        df = claims_df.copy()

        # Apply pre-computed matches
        df["GPI_Match"] = coverage_info[f_name]["gpi_matches"]
        df["Name_Match"] = coverage_info[f_name]["name_matches"]

        # Calculate GPI similarity
        df["GPI_Similarity"] = df["GPI"].apply(
            lambda g: compute_gpi_similarity_fast(
                g, coverage_info[f_name]["prefix_sets"]
            )
        )

        # Continue with the rest of the analysis...
        # (keep the existing analyze_formulary logic from here on)

        # Process results as before
        summ, top10, sim_counts = analyze_formulary_with_coverage(
            f_name, f_df, df, coverage_info[f_name]
        )
        all_summaries.append(summ)
        results[f_name] = {
            "top_disruptions": top10,
            "similarity_counts": sim_counts,
            "coverage": coverage_info[f_name],  # Store coverage for reuse
        }

    return all_summaries, results, coverage_info


# Modified analyze_formulary function that accepts pre-computed coverage
def analyze_formulary_with_coverage(formulary_name, formulary_df, df, coverage_info):
    """
    Modified version of analyze_formulary that uses pre-computed coverage information
    """
    # Use the pre-computed matches
    df["GPI_Match"] = coverage_info["gpi_matches"]
    df["Name_Match"] = coverage_info["name_matches"]

    # Calculate GPI similarity if needed
    if "GPI_Similarity" not in df.columns:
        df["GPI_Similarity"] = df["GPI"].apply(
            lambda g: compute_gpi_similarity_fast(g, coverage_info["prefix_sets"])
        )

    # NEW: Consider claims with GPI similarity of 10+ digits as "covered"
    df["Similar_Enough"] = df["GPI_Similarity"] >= 10

    # Modified: Consider GPI match OR Name match OR Similar Enough for coverage
    df["Covered"] = df["GPI_Match"] | df["Name_Match"] | df["Similar_Enough"]
    df["Status"] = df["Covered"].map({True: "Covered", False: "Disrupted"})

    disrupted = df[df["Status"] == "Disrupted"].copy()

    # For similarity analysis, set default
    df["Max_GPI_Match"] = 0  # Default to no match

    # Set covered claims to special category if exact match or name match
    df.loc[df["GPI_Match"] | df["Name_Match"], "Max_GPI_Match"] = 99

    # For claims covered due to similarity, keep their actual similarity level
    df.loc[
        ~(df["GPI_Match"] | df["Name_Match"]) & df["Similar_Enough"], "Max_GPI_Match"
    ] = df.loc[
        ~(df["GPI_Match"] | df["Name_Match"]) & df["Similar_Enough"], "GPI_Similarity"
    ]

    # For disrupted claims that have name matches
    disrupted_mask = ~df["Covered"]
    df.loc[disrupted_mask & df["Name_Match"], "Max_GPI_Match"] = -1

    # For remaining disrupted claims, keep the similarity level already calculated
    df.loc[disrupted_mask, "Similarity_Level"] = df.loc[
        disrupted_mask, "GPI_Similarity"
    ].apply(lambda x: similarity_labels.get(x, "No close match"))

    # Set similarity level for covered claims
    df.loc[~disrupted_mask, "Similarity_Level"] = df.loc[
        ~disrupted_mask, "Max_GPI_Match"
    ].apply(
        lambda x: (
            "Covered (Exact Match or Name Match)"
            if x == 99
            else (
                "Covered (Same Ingredient)"
                if x >= 10
                else similarity_labels.get(x, "No close match")
            )
        )
    )

    total_claims = df["claimstatus"].sum()
    disrupted_claims = disrupted["claimstatus"].sum()
    disruption_rate = disrupted_claims / total_claims if total_claims > 0 else 0

    # Cost + members (if available)
    disrupted_cost = None
    if "repriced_ingredient_cost" in disrupted.columns:
        disrupted_cost = disrupted["repriced_ingredient_cost"].sum()
    disrupted_members = None
    if "member_id" in disrupted.columns:
        disrupted_members = disrupted["member_id"].nunique()

    summary = {
        "Formulary": formulary_name,
        "Total Claims": total_claims,
        "Disrupted Claims": disrupted_claims,
        "Disruption Rate": disruption_rate,
    }
    if disrupted_cost is not None:
        summary["Disrupted Cost"] = disrupted_cost
    if disrupted_members is not None:
        summary["Disrupted Members"] = disrupted_members

    # --- Build the "Top 10 Disrupted Drugs" Table ---
    # Start with Disrupted Count
    top_disruptions = (
        disrupted.groupby(["GPI", "drugname"])["claimstatus"]
        .sum()
        .reset_index(name="Disrupted Count")
    )

    # If we have member_id, add unique count
    if "member_id" in disrupted.columns:
        top_member = (
            disrupted.groupby(["GPI", "drugname"])["member_id"]
            .nunique()
            .reset_index(name="Members Affected")
        )
        top_disruptions = pd.merge(
            top_disruptions, top_member, on=["GPI", "drugname"], how="left"
        )
    else:
        top_disruptions["Members Affected"] = "N/A"

    # If we have repriced_ingredient_cost, add sum
    if "repriced_ingredient_cost" in disrupted.columns:
        top_cost = (
            disrupted.groupby(["GPI", "drugname"])["repriced_ingredient_cost"]
            .sum()
            .reset_index(name="Drug Cost Affected")
        )
        top_disruptions = pd.merge(
            top_disruptions, top_cost, on=["GPI", "drugname"], how="left"
        )
    else:
        top_disruptions["Drug Cost Affected"] = "N/A"

    # Sort & Keep Top 10
    top_disruptions = top_disruptions.sort_values(
        by="Disrupted Count", ascending=False
    ).head(10)

    # Modified similarity distribution to include all claims
    similarity_counts = df["Similarity_Level"].value_counts().reset_index()
    similarity_counts.columns = ["Similarity Level", "Count"]
    similarity_counts = similarity_counts.sort_values("Count", ascending=False)

    return summary, top_disruptions, similarity_counts


# Replace the existing loop with the optimized analysis
all_summaries, results, coverage_info = analyze_all_formularies(claims, formularies)
summary_df = pd.DataFrame(all_summaries)

# --------------------
# Build Main Tabs
# --------------------
tabs = st.tabs(
    ["🏠 Main Dashboard", "🔄 Formulary Comparison"] + list(formularies.keys())
)

# --------------------
# Main Dashboard (Overview)
# --------------------
with tabs[0]:
    st.header("Overall Disruption Overview")

    # --- 4 Big Metrics (adding member_paid) ---
    col1, col2, col3, col4 = st.columns(4)

    # 1) Total $ Claimed
    if "repriced_ingredient_cost" in claims.columns:
        total_cost = claims["repriced_ingredient_cost"].sum()
        col1.metric("Total $ Claimed", f"${total_cost:,.2f}")
    else:
        col1.metric("Total $ Claimed", "N/A")
        st.warning(
            "No 'repriced_ingredient_cost' column found; cannot compute Total $ Claimed."
        )

    # 2) Total Member Paid (New)
    if "member_paid" in claims.columns:
        total_member_paid = claims["member_paid"].sum()
        col2.metric("Total Member Paid", f"${total_member_paid:,.2f}")
    else:
        col2.metric("Total Member Paid", "N/A")

    # 3) Number of Unique Claimants
    if "member_id" in claims.columns:
        unique_claimants = claims["member_id"].nunique()
        col3.metric("Number of Unique Claimants", f"{unique_claimants:,}")
    else:
        col3.metric("Number of Unique Claimants", "N/A")
        st.warning(
            "No 'member_id' column found; cannot compute Number of Unique Claimants."
        )

    # 4) Total Number of Claims
    if "claimstatus" in claims.columns:
        total_claims_status = claims["claimstatus"].sum()
        col4.metric("Total Number of Claims", f"{total_claims_status:,}")
    else:
        # This warning should not appear anymore since we're creating the column
        col4.metric("Total Number of Claims", f"{total_claims_count:,}")

    st.subheader("Disruption Summary by Formulary")
    # Show summary table
    fmt_dict = {"Disruption Rate": "{:.2%}"}
    if "Disrupted Cost" in summary_df.columns:
        fmt_dict["Disrupted Cost"] = "${:,.2f}"
    if "Disrupted Members" in summary_df.columns:
        fmt_dict["Disrupted Members"] = "{:,.0f}"

    st.dataframe(summary_df.style.format(fmt_dict))

    # Add colored boxes for formulary disruption rates with simplified design and clear labels
    st.markdown(
        """
    <style>
    .metric-box {
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        margin: 10px;
        color: white;
    }
    .metric-box h3 {
        margin: 0 0 5px 0;
        font-size: 1.3em;
        border-bottom: 1px solid rgba(255,255,255,0.3);
        padding-bottom: 5px;
    }
    .stat-label {
        font-size: 0.9em;
        margin: 3px 0 0 0;
        opacity: 0.9;
        font-weight: normal;
    }
    .percentage {
        font-size: 1.8em;
        font-weight: bold;
        margin: 0;
        line-height: 1.2;
    }
    .raw-numbers {
        font-size: 0.8em;
        font-weight: normal;
        opacity: 0.9;
        margin-top: 3px;
    }
    .best {
        background-color: #28a745;
    }
    .premier {
        background-color: #ffd580;  /* Light orange for Premier */
        color: #333;  /* Darker text for light background */
    }
    .premier-plus {
        background-color: #ffa500;  /* Darker orange for Premier Plus */
    }
    .worst {
        background-color: #dc3545;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Define color scheme for formularies
    formulary_colors = {
        "Premier": "premier",
        "Premier Plus": "premier-plus",
        "Preferred": "best",
        "Optimal": "best",
        "Base": "worst",
    }

    # Calculate total cost if available
    total_cost = None
    if "repriced_ingredient_cost" in claims.columns:
        total_cost = claims["repriced_ingredient_cost"].sum()

    # First row - Disruption Rate
    st.markdown("### Claims Disruption Rate")
    disruption_cols = st.columns(5)  # 5 columns for the formularies

    # Second row - Cost Disruption Rate
    st.markdown("### Cost Disruption Rate")
    cost_cols = st.columns(5)  # 5 columns for the formularies

    # Get disruption rates from summary_df
    for i, (idx, row) in enumerate(summary_df.iterrows()):
        # Determine color class based on formulary name
        color_class = formulary_colors.get(row["Formulary"], "worst")

        # Format disruption rate percentage (just the percentage)
        disruption_rate = f"{row['Disruption Rate']:.1%}"

        # Calculate coverage metrics
        total_claims = row["Total Claims"]
        disrupted_claims = row["Disrupted Claims"]

        # Format the "Not Covered" line for claims
        not_covered_display = (
            f"Not Covered: {disrupted_claims:,.0f}/{total_claims:,.0f}"
        )

        # First row: Claims Disruption Rate
        disruption_cols[i].markdown(
            f"""
        <div class="metric-box {color_class}">
            <h3>{row['Formulary']}</h3>
            <div class="stat-label">% of claims that would be disrupted</div>
            <div class="percentage">{disruption_rate}</div>
            <div class="raw-numbers">{not_covered_display}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Second row: Cost Disruption Rate
        if "Disrupted Cost" in row and total_cost is not None and total_cost > 0:
            cost_percent = f"{(row['Disrupted Cost'] / total_cost) * 100:.1f}%"

            # Format the "Not Covered" line for cost
            not_covered_cost_display = (
                f"Not Covered: ${row['Disrupted Cost']:,.0f}/${total_cost:,.0f}"
            )

            cost_cols[i].markdown(
                f"""
            <div class="metric-box {color_class}">
                <h3>{row['Formulary']}</h3>
                <div class="stat-label">% of total cost that would be disrupted</div>
                <div class="percentage">{cost_percent}</div>
                <div class="raw-numbers">{not_covered_cost_display}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            cost_cols[i].markdown(
                f"""
            <div class="metric-box {color_class}">
                <h3>{row['Formulary']}</h3>
                <div class="stat-label">% of total cost that would be disrupted</div>
                <div class="percentage">N/A</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    st.markdown("---")  # Add a separator

    st.subheader("Disruption Comparison")

    # Define consistent colors for the graphs
    color_map = {
        "Premier": "#ffd580",  # Light orange
        "Premier Plus": "#ffa500",  # Darker orange
        "Preferred": "#28a745",  # Green (renamed from "Optimal Brands")
        "Optimal": "#008000",  # Dark green for the new Optimal formulary
        "Base": "#dc3545",  # Red
    }

    # Create two columns for the charts
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        # First chart is now Disruption Rate (moved from right to left)
        summary_df["Disruption Rate %"] = summary_df["Disruption Rate"] * 100
        fig_rate = px.bar(
            summary_df,
            x="Formulary",
            y="Disruption Rate %",
            title="Disruption Rate by Formulary",
            color="Formulary",
            text="Disruption Rate %",
            color_discrete_map=color_map,
        )
        fig_rate.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_rate.update_layout(showlegend=False)
        st.plotly_chart(fig_rate, use_container_width=True)

    with chart_col2:
        # Second chart is now Disrupted Cost
        if "Disrupted Cost" in summary_df.columns:
            fig_cost = px.bar(
                summary_df,
                x="Formulary",
                y="Disrupted Cost",
                title="Disrupted Cost by Formulary",
                color="Formulary",
                text="Disrupted Cost",
                color_discrete_map=color_map,
            )
            # Format the cost display with $ and commas
            fig_cost.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
            fig_cost.update_layout(showlegend=False)
            st.plotly_chart(fig_cost, use_container_width=True)
        else:
            # Display a message if cost data isn't available
            st.info(
                "Cost data is not available to display disrupted cost by formulary."
            )
            st.markdown(
                """
            <div style="height:300px; display:flex; align-items:center; justify-content:center; 
                        border:1px dashed #ccc; border-radius:5px; margin-top:5px;">
                <p style="color:#888; font-style:italic; text-align:center;">
                    No cost data available for visualization.<br>
                    Add 'repriced_ingredient_cost' column to your data to see cost analysis.
                </p>
            </div>
            """,
                unsafe_allow_html=True,
            )

    # Add Top 5 Disruption Tables
    st.subheader("Top 5 Disruptions by Different Metrics")

    col1, col2, col3 = st.columns(3)

    # First, get coverage status for each drug name in each formulary
    drug_coverage = {}
    for f_name, f_df in formularies.items():
        # Extract formulary drug names and create a set for biosimilar checks
        formulary_clean_names = [
            clean_drug_name(desc).lower()
            for desc in f_df["ndcdescription"]
            if isinstance(desc, str)
        ]
        formulary_names_set = set(formulary_clean_names)

        # Pre-check biosimilar coverage once
        has_biosimilar_coverage = any(
            any(biosim in fdrug for biosim in humira_biosimilars_set)
            for fdrug in formulary_names_set
        )

        # Check both GPI and name matches for each drug, including biosimilars
        drug_coverage[f_name] = set()
        for drug in claims["drugname"].unique():
            drug_lower = drug.lower()

            # Check if drug name matches any NDC description
            name_match = any(drug_lower in desc for desc in formulary_names_set)

            # Check biosimilar coverage (only if needed)
            if not name_match:
                is_biosimilar = any(
                    biosim in drug_lower for biosim in humira_biosimilars_set
                )
                if is_biosimilar and has_biosimilar_coverage:
                    name_match = True

            # Check if any GPI matches
            gpi_match = any(claims[claims["drugname"] == drug]["GPI"].isin(f_df["GPI"]))

            if name_match or gpi_match:
                drug_coverage[f_name].add(drug)

    # Find drugs that are not covered in ANY formulary
    all_drugs = set(claims["drugname"].unique())
    drugs_covered_somewhere = set().union(*drug_coverage.values())
    fully_disrupted_drugs = all_drugs - drugs_covered_somewhere

    # Filter claims to only those disrupted across all formularies
    all_disrupted = claims[claims["drugname"].isin(fully_disrupted_drugs)]

    with col1:
        st.markdown("**Top 5 by Drug Cost (Disrupted in All Formularies)**")
        if "repriced_ingredient_cost" in claims.columns:
            cost_top5 = (
                all_disrupted.groupby("drugname")["repriced_ingredient_cost"]
                .sum()
                .sort_values(ascending=False)
                .head(5)
                .reset_index()
            )
            cost_top5.columns = ["Drug Name", "Disrupted Cost"]
            st.dataframe(cost_top5.style.format({"Disrupted Cost": "${:,.2f}"}))
        else:
            st.info("Cost data not available")

    with col2:
        st.markdown("**Top 5 by Claim Count (Disrupted in All Formularies)**")
        claims_top5 = (
            all_disrupted.groupby("drugname")["claimstatus"]
            .sum()
            .sort_values(ascending=False)
            .head(5)
            .reset_index()
        )
        claims_top5.columns = ["Drug Name", "Disrupted Claims"]
        st.dataframe(claims_top5)

    with col3:
        st.markdown("**Top 5 by Members Affected (Disrupted in All Formularies)**")
        if "member_id" in claims.columns:
            members_top5 = (
                all_disrupted.groupby("drugname")["member_id"]
                .nunique()
                .sort_values(ascending=False)
                .head(5)
                .reset_index()
            )
            members_top5.columns = ["Drug Name", "Members Affected"]
            st.dataframe(members_top5)
        else:
            st.info("Member data not available")


# --------------------
# Formulary Comparison Tab
# --------------------
with tabs[1]:
    st.header("Formulary Comparison")

    # Create search interface
    col1, col2 = st.columns([1, 2])
    with col1:
        search_type = st.selectbox(
            "Search by:", ["Drug Name", "GPI"], key="comparison_search_type"
        )
    with col2:
        search_term = st.text_input(
            "Enter search term:",
            key="comparison_search_term",
            placeholder="Enter drug name or GPI...",
        )

    if search_term:
        # Create comparison DataFrame
        comparison_data = []

        # Get all unique drugs/GPIs from claims
        if search_type == "Drug Name":
            search_term = search_term.lower()
            matching_drugs = claims[
                claims["cleaned_drugname"].str.contains(search_term, na=False)
            ][["drugname", "cleaned_drugname", "GPI"]].drop_duplicates()
        else:  # GPI search
            matching_drugs = claims[claims["GPI"].str.contains(search_term, na=False)][
                ["drugname", "cleaned_drugname", "GPI"]
            ].drop_duplicates()

        # For each matching drug, check coverage in each formulary
        for _, row in matching_drugs.iterrows():
            drug_info = {"Drug Name": row["drugname"], "GPI": row["GPI"]}

            # Check coverage in each formulary
            for f_name, f_df in formularies.items():
                # Extract all drug names in formulary for biosimilar checks
                formulary_drug_names = [
                    clean_drug_name(desc).lower()
                    for desc in f_df["ndcdescription"]
                    if isinstance(desc, str)
                ]

                # Check both GPI and name matches with biosimilar logic
                gpi_match = row["GPI"] in f_df["GPI"].values
                name_match = any(
                    row["cleaned_drugname"] in clean_drug_name(desc).lower()
                    for desc in f_df["ndcdescription"]
                    if isinstance(desc, str)
                ) or is_biosimilar_covered_optimized(
                    row["cleaned_drugname"], set(formulary_drug_names)
                )

                drug_info[f_name] = "✅" if (gpi_match or name_match) else "❌"

            comparison_data.append(drug_info)

        if comparison_data:
            # Create and display comparison table
            comparison_df = pd.DataFrame(comparison_data)

            # Reorder columns to put Drug Name and GPI first
            cols = ["Drug Name", "GPI"] + [
                col for col in comparison_df.columns if col not in ["Drug Name", "GPI"]
            ]
            comparison_df = comparison_df[cols]

            # Display the table with custom formatting
            st.markdown("### Coverage Comparison")
            st.markdown("✅ = Covered, ❌ = Not Covered")

            # Function to highlight covered/not covered
            def highlight_coverage(val):
                if val == "✅":
                    return "background-color: #e6ffe6"
                elif val == "❌":
                    return "background-color: #ffe6e6"
                return ""

            # Apply the styling
            styled_df = comparison_df.style.applymap(
                highlight_coverage,
                subset=[
                    col
                    for col in comparison_df.columns
                    if col not in ["Drug Name", "GPI"]
                ],
            )

            st.dataframe(
                styled_df, height=400, hide_index=True, use_container_width=True
            )

            # Add alternatives section
            st.markdown("### Similar Covered Alternatives")

            # For each drug in the search results
            for _, drug_row in matching_drugs.iterrows():
                drug_name = drug_row["drugname"]
                drug_gpi = drug_row["GPI"]

                st.markdown(f"**Alternatives for {drug_name} (GPI: {drug_gpi})**")

                # Create columns for each formulary
                form_cols = st.columns(len(formularies))

                # For each formulary, find similar covered drugs
                for col, (f_name, f_df) in zip(form_cols, formularies.items()):
                    with col:
                        st.markdown(f"***{f_name}***")

                        # Get all covered drugs in this formulary
                        covered_drugs = f_df[["ndcdescription", "GPI"]].copy()
                        covered_drugs["clean_name"] = covered_drugs[
                            "ndcdescription"
                        ].apply(clean_drug_name)
                        covered_drugs = covered_drugs.drop_duplicates(subset=["GPI"])

                        # Remove the search drug
                        covered_drugs = covered_drugs[
                            covered_drugs["clean_name"].str.lower() != drug_name.lower()
                        ]

                        if len(covered_drugs) > 0:
                            # Calculate name similarity first
                            covered_drugs["name_match"] = (
                                covered_drugs["clean_name"]
                                .str.lower()
                                .str.contains(drug_name.lower(), na=False)
                            )

                            # Calculate GPI similarity scores
                            covered_drugs["gpi_similarity"] = covered_drugs[
                                "GPI"
                            ].apply(
                                lambda x: sum(
                                    1
                                    for i, (a, b) in enumerate(
                                        zip(x.zfill(14), drug_gpi.zfill(14))
                                    )
                                    if a == b
                                    and all(
                                        x.zfill(14)[j] == drug_gpi.zfill(14)[j]
                                        for j in range(i)
                                    )
                                )
                            )

                            # Sort by name match first, then GPI similarity
                            alternatives = covered_drugs.sort_values(
                                ["name_match", "gpi_similarity", "ndcdescription"],
                                ascending=[False, False, True],
                            ).head(5)

                            # Create a formatted display of alternatives
                            for _, alt in alternatives.iterrows():
                                matching_digits = alt["gpi_similarity"]

                                # Skip if less than therapeutic class match and no name match
                                if matching_digits < 6 and not alt["name_match"]:
                                    continue

                                if alt["name_match"]:
                                    similarity = "💫 NDC Description Match"
                                elif matching_digits >= 12:
                                    similarity = "🟢 Same ingredient"
                                elif matching_digits >= 10:
                                    similarity = "🟡 Similar ingredient"
                                elif matching_digits >= 8:
                                    similarity = "🟠 Same subclass"
                                else:
                                    similarity = "🔴 Same class"

                                st.markdown(
                                    f"{similarity}  \n"
                                    f"**{alt['ndcdescription']}**  \n"
                                    f"Clean name: {alt['clean_name']}  \n"
                                    f"GPI: {alt['GPI']}  \n"
                                    f"*({matching_digits} digits match)*"
                                )
                        else:
                            st.markdown("*No alternatives found*")

                st.markdown("---")  # Separator between drugs

        else:
            st.info(f"No drugs found matching the search term: {search_term}")
    else:
        st.info("Enter a search term to compare coverage across formularies")


# --------------------
# Individual Formulary Tabs
# --------------------
for i, f_name in enumerate(formularies.keys(), start=2):
    with tabs[i]:
        st.header(f"Details for {f_name}")
        sim_counts_df = results[f_name]["similarity_counts"]

        # ---- Top 5 Disruption Tables ----
        st.subheader("Top 5 Disruptions by Different Metrics")

        col1, col2, col3 = st.columns(3)

        # Use pre-computed coverage instead of copying claims again
        # Get disrupted claims for this formulary using stored results
        df = claims.copy()  # We still need one copy here
        df["GPI_Match"] = coverage_info[f_name]["gpi_matches"]
        df["Name_Match"] = coverage_info[f_name]["name_matches"]
        df["Covered"] = df["GPI_Match"] | df["Name_Match"]

        disrupted = df[~df["Covered"]]

        with col1:
            st.markdown("**Top 5 by Drug Cost**")
            if "repriced_ingredient_cost" in claims.columns:
                cost_top5 = (
                    disrupted.groupby("drugname")["repriced_ingredient_cost"]
                    .sum()
                    .sort_values(ascending=False)
                    .head(5)
                    .reset_index()
                )
                cost_top5.columns = ["Drug Name", "Disrupted Cost"]
                st.dataframe(cost_top5.style.format({"Disrupted Cost": "${:,.2f}"}))
            else:
                st.info("Cost data not available")

        with col2:
            st.markdown("**Top 5 by Claim Count**")
            claims_top5 = (
                disrupted.groupby("drugname")["claimstatus"]
                .sum()
                .sort_values(ascending=False)
                .head(5)
                .reset_index()
            )
            claims_top5.columns = ["Drug Name", "Disrupted Claims"]
            st.dataframe(claims_top5)

        with col3:
            st.markdown("**Top 5 by Members Affected**")
            if "member_id" in claims.columns:
                members_top5 = (
                    disrupted.groupby("drugname")["member_id"]
                    .nunique()
                    .sort_values(ascending=False)
                    .head(5)
                    .reset_index()
                )
                members_top5.columns = ["Drug Name", "Members Affected"]
                st.dataframe(members_top5)
            else:
                st.info("Member data not available")

        # ---- Pie Chart: Disruption by Similarity Level ----
        st.subheader("Disruption by Similarity Level")
        fig_sim = px.pie(
            sim_counts_df,
            names="Similarity Level",
            values="Count",
            title="All Claims by Coverage/Similarity Level",
        )
        st.plotly_chart(fig_sim, use_container_width=True, key=f"sim_pie_{f_name}")

        # ---- GPI Similarity Legend ----
        st.markdown(
            """
            **GPI Similarity Legend**  
            - **14 digits**: Exact match (Same product, strength, and form)  
            - **12 digits**: Same ingredient, route, and dosage form  
            - **10 digits**: Same ingredient, different strength/form  
            - **8 digits**: Same subclass (similar therapeutic intent)  
            - **6 digits**: Same class (broad therapeutic category)  
            - **-1 digits**: NDC Description match only  
            - **0 digits**: No close match found  
            """
        )

        # ---- Detailed Exclusion Table (Moved to bottom) ----
        st.markdown("---")  # Add a visual separator
        st.subheader("Detailed Exclusion Status")

        # Add dropdown for grouping selection
        col1, col2 = st.columns([2, 3])
        with col1:
            group_by = st.selectbox(
                "Group by:", ["Drug Name", "NDC"], key=f"group_select_{f_name}"
            )
        with col2:
            search_term = st.text_input(
                "Search:",
                key=f"search_{f_name}",
                placeholder="Enter drug name or NDC...",
            )

        # Prepare detailed exclusion data
        df = claims.copy()
        df["GPI_Match"] = df["GPI"].isin(formularies[f_name]["GPI"])

        # Add Name_Match calculation
        df["Name_Match"] = df["cleaned_drugname"].apply(
            lambda x: any(
                x in desc.lower()
                for desc in formularies[f_name]["ndcdescription"]
                if isinstance(desc, str)
            )
        )

        # Consider both GPI match OR Name match for coverage
        df["Covered"] = df["GPI_Match"] | df["Name_Match"]

        # Group by selected field
        group_field = "drugname" if group_by == "Drug Name" else "ndc"

        # Modified aggregation to avoid column name conflicts
        if group_by == "Drug Name":
            detailed_exclusions = (
                df.groupby(group_field)
                .agg(
                    {
                        "Covered": lambda x: "Included" if any(x) else "Excluded",
                        "GPI_Match": "any",  # Add this to see GPI matches
                        "Name_Match": "any",  # Add this to see Name matches
                        "ndc": "nunique",  # Count unique NDCs
                        "therapeuticclass02": "first",  # Add therapeutic class
                        "claimstatus": "sum",  # Add claim count for cost per claim calculation
                        "repriced_ingredient_cost": (
                            "sum"
                            if "repriced_ingredient_cost" in df.columns
                            else lambda x: None
                        ),
                        "member_id": (
                            "nunique" if "member_id" in df.columns else lambda x: None
                        ),
                    }
                )
                .reset_index()
            )
            # Rename columns
            detailed_exclusions.columns = [
                "Drug Name",
                "Status",
                "GPI Match",
                "Name Match",
                "NDC Count",
                "Therapeutic Class",
                "Claim Count",
                "Total Cost",
                "Unique Members",
            ]

            # Calculate cost per claim
            if (
                "Total Cost" in detailed_exclusions.columns
                and "Claim Count" in detailed_exclusions.columns
            ):
                detailed_exclusions["Cost per Claim"] = (
                    detailed_exclusions["Total Cost"]
                    / detailed_exclusions["Claim Count"]
                )
                detailed_exclusions.drop("Total Cost", axis=1, inplace=True)

        else:  # NDC view
            detailed_exclusions = (
                df.groupby(group_field)
                .agg(
                    {
                        "Covered": lambda x: "Included" if any(x) else "Excluded",
                        "GPI_Match": "any",  # Add this to see GPI matches
                        "Name_Match": "any",  # Add this to see Name matches
                        "drugname": "first",
                        "therapeuticclass02": "first",  # Add therapeutic class
                        "claimstatus": "sum",  # Add claim count for cost per claim calculation
                        "repriced_ingredient_cost": (
                            "sum"
                            if "repriced_ingredient_cost" in df.columns
                            else lambda x: None
                        ),
                        "member_id": (
                            "nunique" if "member_id" in df.columns else lambda x: None
                        ),
                    }
                )
                .reset_index()
            )
            # Rename columns
            detailed_exclusions.columns = [
                "NDC",
                "Status",
                "GPI Match",
                "Name Match",
                "Drug Name",
                "Therapeutic Class",
                "Claim Count",
                "Total Cost",
                "Unique Members",
            ]

            # Calculate cost per claim
            if (
                "Total Cost" in detailed_exclusions.columns
                and "Claim Count" in detailed_exclusions.columns
            ):
                detailed_exclusions["Cost per Claim"] = (
                    detailed_exclusions["Total Cost"]
                    / detailed_exclusions["Claim Count"]
                )
                detailed_exclusions.drop("Total Cost", axis=1, inplace=True)

        # Apply search filter
        if search_term:
            search_term = search_term.lower()
            if group_by == "Drug Name":
                mask = (
                    detailed_exclusions["Drug Name"]
                    .str.lower()
                    .str.contains(search_term, na=False)
                )
            else:  # NDC view
                mask = detailed_exclusions["NDC"].str.lower().str.contains(
                    search_term, na=False
                ) | detailed_exclusions["Drug Name"].str.lower().str.contains(
                    search_term, na=False
                )
            detailed_exclusions = detailed_exclusions[mask]

        # Format and display the table
        fmt_dict = {}
        if "Cost per Claim" in detailed_exclusions.columns:
            fmt_dict["Cost per Claim"] = "${:,.2f}"

        def highlight_status(row):
            color = "background-color: "
            color += "#e6ffe6" if row["Status"] == "Included" else "#ffe6e6"
            return [color] * len(row)

        st.dataframe(
            detailed_exclusions.style.apply(highlight_status, axis=1).format(fmt_dict),
            height=400,
            hide_index=True,
            use_container_width=True,  # Make table use full width
        )
