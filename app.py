import pandas as pd
import plotly.express as px
import streamlit as st
import re


def clean_drug_name(name):
    """Remove text in parentheses, 'ER', 'HCl', and 'HFA' from drug names."""
    # First remove text in parentheses
    name = re.sub(r"\s*\([^)]*\)", "", name)
    # Remove 'ER', 'HCl', and 'HFA' (with or without spaces)
    name = re.sub(r"\s*ER\b|\s*HCl\b|\s*HFA\b", "", name)
    # Clean up any extra whitespace and return
    return name.strip()


# --------------------
# Page Config
# --------------------
st.set_page_config(
    layout="wide", page_title="Essential & Optimal Formulary Disruption Dashboard"
)

# --------------------
# Load Formulary Data
# --------------------
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
    "Optimal Brands": pd.read_excel(
        formulary_excel, sheet_name="OptimalBrands Formulary", dtype={"GPI": str}
    ),
}

# Apply formulary inheritance
formularies = {}
for name, df in formularies_raw.items():
    df["GPI"] = df["GPI"].astype(str)
    if name == "Base":
        formularies[name] = df
    elif name == "Premier":
        # Combine Base and Premier
        formularies[name] = pd.concat([formularies["Base"], df]).drop_duplicates()
    elif name == "Premier Plus":
        # Combine Base, Premier, and Premier Plus
        formularies[name] = pd.concat(
            [formularies["Base"], formularies["Premier"], df]
        ).drop_duplicates()
    elif name == "Optimal Brands":
        # Combine Base and Optimal Brands
        formularies[name] = pd.concat([formularies["Base"], df]).drop_duplicates()

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

uploaded_file = st.file_uploader("📁 Upload your claims file (CSV)", type=["csv"])

if not uploaded_file:
    st.info("👈 Please upload a claims file to begin.")
    st.stop()

# Load Claims
claims = pd.read_csv(uploaded_file, dtype={"GPI": str, "NDC": str})
required_cols = ["NDC"]
missing_req = [col for col in required_cols if col not in claims.columns]
if missing_req:
    st.error(f"❌ Uploaded file must contain columns: {missing_req}")
    st.stop()

claims = claims[claims["NDC"].notna()]
claims = claims.rename(columns={"NDC": "ndc"})
claims["ndc"] = claims["ndc"].apply(lambda x: x.zfill(11))

# load Medispan
medispan = pd.read_parquet("./data/medispan.parquet")

# merge claims with medispan
claims = claims.merge(medispan, on="ndc", how="left")

# rename gpi to GPI
claims = claims.rename(columns={"gpi": "GPI"})

claims["GPI"] = claims["GPI"].astype(str)
claims["drugname"] = claims["drugname"].astype(str).apply(clean_drug_name)


# --------------------
# Similarity Logic
# --------------------
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
# Analysis Function
# --------------------
def analyze_formulary(formulary_name, formulary_df):
    """
    Analyzes how many claims are disrupted vs. covered,
    calculates similarity level for disrupted claims,
    and returns summary data + top 10 disruptions + similarity distribution.
    """
    df = claims.copy()

    # First check GPI match
    df["GPI_Match"] = df["GPI"].isin(formulary_df["GPI"])

    # Check NDC description match
    df["Name_Match"] = (
        df["drugname"]
        .str.lower()
        .apply(
            lambda x: any(
                x in desc.lower()
                for desc in formulary_df["ndcdescription"]
                if isinstance(desc, str)
            )
        )
    )

    # Modified: Consider both GPI match OR Name match for coverage
    df["Covered"] = df["GPI_Match"] | df["Name_Match"]
    df["Status"] = df["Covered"].map({True: "Covered", False: "Disrupted"})

    disrupted = df[df["Status"] == "Disrupted"].copy()

    # Modified to include both covered and disrupted claims in similarity analysis
    df["Max_GPI_Match"] = 0  # Default to no match

    # Set covered claims to special category
    df.loc[df["Covered"], "Max_GPI_Match"] = 99

    # For disrupted claims, calculate similarity
    disrupted_mask = ~df["Covered"]
    df.loc[disrupted_mask & df["Name_Match"], "Max_GPI_Match"] = -1

    # For remaining disrupted claims, calculate GPI similarity
    remaining_disrupted = df[disrupted_mask & ~df["Name_Match"]]
    if len(remaining_disrupted) > 0:
        df.loc[remaining_disrupted.index, "Max_GPI_Match"] = remaining_disrupted[
            "GPI"
        ].apply(lambda g: compute_gpi_similarity(g, formulary_df["GPI"].tolist()))

    df["Similarity_Level"] = df["Max_GPI_Match"].map(similarity_labels)

    total_claims = len(df)
    disrupted_claims = len(disrupted)
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
        disrupted.groupby(["GPI", "drugname"])
        .size()
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


# --------------------
# Analyze All Formularies
# --------------------
all_summaries = []
results = {}

for f_name, f_df in formularies.items():
    summ, top10, sim_counts = analyze_formulary(f_name, f_df)
    all_summaries.append(summ)
    results[f_name] = {"top_disruptions": top10, "similarity_counts": sim_counts}

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
    try:
        if "repriced_ingredient_cost" in claims.columns:
            total_cost = claims["repriced_ingredient_cost"].sum()
            col1.metric("Total $ Claimed", f"${total_cost:,.2f}")
        else:
            col1.metric("Total $ Claimed", "N/A")
            st.info("💡 Add 'repriced_ingredient_cost' column to see total claim costs")
    except Exception as e:
        st.error(
            "❌ Error calculating Total $ Claimed. Please ensure 'repriced_ingredient_cost' contains valid numeric values."
        )

    # 2) Total Member Paid
    try:
        if "member_paid" in claims.columns:
            total_member_paid = claims["member_paid"].sum()
            col2.metric("Total Member Paid", f"${total_member_paid:,.2f}")
        else:
            col2.metric("Total Member Paid", "N/A")
            st.info("💡 Add 'member_paid' column to see total member payments")
    except Exception as e:
        st.error(
            "❌ Error calculating Total Member Paid. Please ensure 'member_paid' contains valid numeric values."
        )

    # 3) Number of Unique Claimants
    try:
        if "member_id" in claims.columns:
            unique_claimants = claims["member_id"].nunique()
            col3.metric("Number of Unique Claimants", f"{unique_claimants:,}")
        else:
            col3.metric("Number of Unique Claimants", "N/A")
            st.info("💡 Add 'member_id' column to see unique claimant counts")
    except Exception as e:
        st.error(
            "❌ Error calculating Unique Claimants. Please ensure 'member_id' column is properly formatted."
        )

    # 4) Total Number of Claims
    try:
        if "claimstatus" in claims.columns:
            total_claims_status = claims["claimstatus"].sum()
            col4.metric("Total Number of Claims", f"{total_claims_status:,}")
        else:
            col4.metric("Total Number of Claims", len(claims))  # Use total rows instead
            st.info("💡 Add 'claimstatus' column for more accurate claim counting")
    except Exception as e:
        st.error(
            "❌ Error calculating Total Claims. Please ensure 'claimstatus' contains valid numeric values."
        )

    st.subheader("Disruption Summary by Formulary")
    # Show summary table
    fmt_dict = {"Disruption Rate": "{:.2%}"}
    if "Disrupted Cost" in summary_df.columns:
        fmt_dict["Disrupted Cost"] = "${:,.2f}"
    if "Disrupted Members" in summary_df.columns:
        fmt_dict["Disrupted Members"] = "{:,.0f}"

    st.dataframe(summary_df.style.format(fmt_dict))

    # Add colored boxes for formulary disruption rates
    st.markdown(
        """
    <style>
    .metric-box {
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 10px;
        color: white;
    }
    .metric-box h3 {
        margin: 0;
        font-size: 1.5em;
    }
    .metric-box p {
        margin: 5px 0 0 0;
        font-size: 2em;
        font-weight: bold;
    }
    .best {
        background-color: #28a745;
    }
    .premier {
        background-color: #ffd580;  /* Light orange for Premier */
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

    # Create a row of 4 columns for the formulary metrics
    form_col1, form_col2, form_col3, form_col4 = st.columns(4)

    # Sort disruption rates to determine coloring
    sorted_rates = summary_df.sort_values("Disruption Rate")
    best_rate = sorted_rates.iloc[0]["Disruption Rate"]
    worst_rate = sorted_rates.iloc[-1]["Disruption Rate"]

    # Define color scheme for formularies
    formulary_colors = {
        "Premier": "premier",
        "Premier Plus": "premier-plus",
        "Optimal Brands": "best",  # Changed to use green for Optimal Brands
        "Base": "worst",  # Changed to use red for Base
    }

    # Get disruption rates from summary_df
    for col, (idx, row) in zip(
        [form_col1, form_col2, form_col3, form_col4], summary_df.iterrows()
    ):
        # Determine color class based on formulary name
        color_class = formulary_colors.get(
            row["Formulary"], "worst"
        )  # Default to red if not found

        col.markdown(
            f"""
        <div class="metric-box {color_class}">
            <h3>{row['Formulary']}</h3>
            <p>{row['Disruption Rate']:.1%}</p>
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
        "Optimal Brands": "#28a745",  # Green
        "Base": "#dc3545",  # Red
    }

    # Create two columns for the charts
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        # Claims bar chart
        fig_claims = px.bar(
            summary_df,
            x="Formulary",
            y="Disrupted Claims",
            title="Disrupted Claims by Formulary",
            color="Formulary",
            text="Disrupted Claims",
            color_discrete_map=color_map,
        )
        fig_claims.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
        fig_claims.update_layout(showlegend=False)
        st.plotly_chart(fig_claims, use_container_width=True)

    with chart_col2:
        # Rate bar chart
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

    # Add Top 5 Disruption Tables
    st.subheader("Top 5 Disruptions by Different Metrics")

    col1, col2, col3 = st.columns(3)

    # Get claims that are disrupted across ALL formularies
    df = claims.copy()

    # First, get coverage status for each drug name in each formulary
    drug_coverage = {}
    for f_name, f_df in formularies.items():
        # Check both GPI and name matches for each drug
        drug_coverage[f_name] = set()
        for drug in df["drugname"].unique():
            # Check if drug name matches any NDC description
            name_match = any(
                drug.lower() in desc.lower()
                for desc in f_df["ndcdescription"]
                if isinstance(desc, str)
            )
            # Check if any GPI matches
            gpi_match = any(df[df["drugname"] == drug]["GPI"].isin(f_df["GPI"]))
            if name_match or gpi_match:
                drug_coverage[f_name].add(drug)

    # Find drugs that are not covered in ANY formulary
    all_drugs = set(df["drugname"].unique())
    drugs_covered_somewhere = set().union(*drug_coverage.values())
    fully_disrupted_drugs = all_drugs - drugs_covered_somewhere

    # Filter claims to only those disrupted across all formularies
    all_disrupted = df[df["drugname"].isin(fully_disrupted_drugs)]

    with col1:
        st.markdown("**Top 5 by Drug Cost (Disrupted in All Formularies)**")
        try:
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
                st.info(
                    "💡 Cost data not available. Add 'repriced_ingredient_cost' column to see cost analysis."
                )
        except Exception as e:
            st.error(
                "❌ Error generating cost analysis. Please ensure 'repriced_ingredient_cost' contains valid numeric values."
            )

    with col2:
        st.markdown("**Top 5 by Claim Count (Disrupted in All Formularies)**")
        try:
            claims_top5 = (
                all_disrupted.groupby("drugname")
                .size()
                .sort_values(ascending=False)
                .head(5)
                .reset_index()
            )
            claims_top5.columns = ["Drug Name", "Disrupted Claims"]
            st.dataframe(claims_top5)
        except Exception as e:
            st.error(
                "❌ Error generating claims analysis. Please check your data format."
            )

    with col3:
        st.markdown("**Top 5 by Members Affected (Disrupted in All Formularies)**")
        try:
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
                st.info(
                    "💡 Member data not available. Add 'member_id' column to see member analysis."
                )
        except Exception as e:
            st.error(
                "❌ Error generating member analysis. Please ensure 'member_id' column is properly formatted."
            )


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
                claims["drugname"].str.lower().str.contains(search_term, na=False)
            ][["drugname", "GPI"]].drop_duplicates()
        else:  # GPI search
            matching_drugs = claims[claims["GPI"].str.contains(search_term, na=False)][
                ["drugname", "GPI"]
            ].drop_duplicates()

        # For each matching drug, check coverage in each formulary
        for _, row in matching_drugs.iterrows():
            drug_info = {"Drug Name": row["drugname"], "GPI": row["GPI"]}

            # Check coverage in each formulary
            for f_name, f_df in formularies.items():
                # Check both GPI and name matches
                gpi_match = row["GPI"] in f_df["GPI"].values
                name_match = any(
                    row["drugname"].lower() in clean_drug_name(desc).lower()
                    for desc in f_df["ndcdescription"]
                    if isinstance(desc, str)
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

        # Get disrupted claims for this formulary
        df = claims.copy()
        df["GPI_Match"] = df["GPI"].isin(formularies[f_name]["GPI"])

        # Add Name_Match calculation
        df["Name_Match"] = (
            df["drugname"]
            .str.lower()
            .apply(
                lambda x: any(
                    x in desc.lower()
                    for desc in formularies[f_name]["ndcdescription"]
                    if isinstance(desc, str)
                )
            )
        )

        # Consider both GPI match OR Name match for coverage
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
                disrupted.groupby("drugname")
                .size()
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
        df["Name_Match"] = (
            df["drugname"]
            .str.lower()
            .apply(
                lambda x: any(
                    x in desc.lower()
                    for desc in formularies[f_name]["ndcdescription"]
                    if isinstance(desc, str)
                )
            )
        )

        # Consider both GPI match OR Name match for coverage
        df["Covered"] = df["GPI_Match"] | df["Name_Match"]

        # Group by selected field
        group_field = "drugname" if group_by == "Drug Name" else "ndc"

        # Modified aggregation to avoid column name conflicts
        if group_by == "Drug Name":
            # Build aggregation dictionary dynamically based on available columns
            agg_dict = {
                "Covered": lambda x: "Included" if any(x) else "Excluded",
                "GPI_Match": "any",  # Add this to see GPI matches
                "Name_Match": "any",  # Add this to see Name matches
                "ndc": "nunique",  # Count unique NDCs
            }

            # Add optional columns if they exist
            if "repriced_ingredient_cost" in df.columns:
                agg_dict["repriced_ingredient_cost"] = "sum"
            if "member_id" in df.columns:
                agg_dict["member_id"] = "nunique"

            detailed_exclusions = df.groupby(group_field).agg(agg_dict).reset_index()

            # Rename columns and add placeholders for missing columns
            column_mapping = {
                group_field: "Drug Name",
                "Covered": "Status",
                "GPI_Match": "GPI Match",
                "Name_Match": "Name Match",
                "ndc": "NDC Count",
            }

            if "repriced_ingredient_cost" in df.columns:
                column_mapping["repriced_ingredient_cost"] = "Total Cost"
            else:
                detailed_exclusions["Total Cost"] = None

            if "member_id" in df.columns:
                column_mapping["member_id"] = "Unique Members"
            else:
                detailed_exclusions["Unique Members"] = None

            detailed_exclusions.columns = [
                column_mapping.get(col, col) for col in detailed_exclusions.columns
            ]

        else:  # NDC view
            # Build aggregation dictionary dynamically based on available columns
            agg_dict = {
                "Covered": lambda x: "Included" if any(x) else "Excluded",
                "GPI_Match": "any",  # Add this to see GPI matches
                "Name_Match": "any",  # Add this to see Name matches
                "drugname": "first",
            }

            # Add optional columns if they exist
            if "repriced_ingredient_cost" in df.columns:
                agg_dict["repriced_ingredient_cost"] = "sum"
            if "member_id" in df.columns:
                agg_dict["member_id"] = "nunique"

            detailed_exclusions = df.groupby(group_field).agg(agg_dict).reset_index()

            # Rename columns and add placeholders for missing columns
            column_mapping = {
                group_field: "NDC",
                "Covered": "Status",
                "GPI_Match": "GPI Match",
                "Name_Match": "Name Match",
                "drugname": "Drug Name",
            }

            if "repriced_ingredient_cost" in df.columns:
                column_mapping["repriced_ingredient_cost"] = "Total Cost"
            else:
                detailed_exclusions["Total Cost"] = None

            if "member_id" in df.columns:
                column_mapping["member_id"] = "Unique Members"
            else:
                detailed_exclusions["Unique Members"] = None

            detailed_exclusions.columns = [
                column_mapping.get(col, col) for col in detailed_exclusions.columns
            ]

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
        try:
            fmt_dict = {}
            if "Total Cost" in detailed_exclusions.columns:

                def currency_format(x):
                    if pd.isna(x) or x is None or x == "N/A":
                        return "N/A"
                    try:
                        return f"${float(x):,.2f}"
                    except (ValueError, TypeError):
                        return str(x)

                fmt_dict["Total Cost"] = currency_format

            if "Unique Members" in detailed_exclusions.columns:

                def member_format(x):
                    if pd.isna(x) or x is None or x == "N/A":
                        return "N/A"
                    try:
                        return f"{int(x):,}"
                    except (ValueError, TypeError):
                        return str(x)

                fmt_dict["Unique Members"] = member_format

            def highlight_status(row):
                color = "background-color: "
                color += "#e6ffe6" if row["Status"] == "Included" else "#ffe6e6"
                return [color] * len(row)

            display_df = detailed_exclusions.copy()
            display_df = display_df.fillna("N/A")

            st.dataframe(
                display_df.style.apply(highlight_status, axis=1).format(fmt_dict),
                height=400,
                hide_index=True,
                use_container_width=True,
            )
        except Exception as e:
            st.error(
                "❌ Error generating detailed exclusion table. Please check your data format and ensure all required columns are present."
            )
            st.info("💡 Required columns: Drug Name/NDC, Status, GPI Match, Name Match")
