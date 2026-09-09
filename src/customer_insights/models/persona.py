"""
Business Persona mapping and Prescriptive Marketing Action Engine.
Transforms mathematical clusters and RFM scores into actionable customer archetypes and playbooks.
"""

from typing import Dict, Any, List, Optional
import pandas as pd


PERSONA_PLAYBOOKS = {
    "Champions (VIPs)": {
        "description": "High frequency, high monetary spend, and recent engagement. Brand advocates.",
        "marketing_strategy": "Reward loyalty with exclusive early access, VIP beta programs, and personal perks.",
        "discount_incentive": "0% - 5% (Focus on exclusivity rather than deep discounts)",
        "preferred_channel": "Dedicated VIP Email & Personal Concierge",
        "priority_level": "High (Preserve & Delight)"
    },
    "Loyal Customers": {
        "description": "Steady buying habits, responsive to promotions, and good lifetime value.",
        "marketing_strategy": "Upsell higher-tier products and cross-sell accessories in related categories.",
        "discount_incentive": "10% on bundles or loyalty point multipliers",
        "preferred_channel": "Targeted Email Newsletter & In-App Banners",
        "priority_level": "High (Nurture & Expand)"
    },
    "Potential Loyalists": {
        "description": "Recent buyers with promising engagement and above-average spend.",
        "marketing_strategy": "Offer membership/loyalty program onboarding and recommend category bestsellers.",
        "discount_incentive": "10% - 15% on second/third purchase",
        "preferred_channel": "Automated Onboarding Email Drip",
        "priority_level": "Medium (Activate)"
    },
    "At-Risk Customers": {
        "description": "High past value but long recency without purchases. Danger of permanent churn.",
        "marketing_strategy": "Urgent win-back campaigns, personalized reactivation discounts, and satisfaction surveys.",
        "discount_incentive": "20% - 25% Time-limited reactivation coupon",
        "preferred_channel": "SMS Alert + Re-engagement Email Series",
        "priority_level": "Critical (Win Back Immediately)"
    },
    "Hibernating / Casuals": {
        "description": "Low frequency and long inactivity. Price-sensitive or one-off purchasers.",
        "marketing_strategy": "Automated low-cost liquidation campaigns or seasonal clearance alerts.",
        "discount_incentive": "Standard clearance promos (avoid expensive direct sales outreach)",
        "preferred_channel": "Mass Marketing / Retargeting Ads",
        "priority_level": "Low (Low Touch)"
    },
    "Critical Reviewers": {
        "description": "Active buyers with below-average ratings (< 3.0 stars). High churn and negative word-of-mouth risk.",
        "marketing_strategy": "Proactive customer success contact to resolve friction and collect feedback.",
        "discount_incentive": "Apology credit / Replacement support voucher",
        "preferred_channel": "Direct Customer Support Outreach",
        "priority_level": "High (Service Recovery)"
    }
}


class PersonaEngine:
    """
    Classifies customers into business personas and prescribes targeted marketing actions.
    """

    @staticmethod
    def assign_persona(
        recency: float,
        frequency: float,
        monetary: float,
        avg_rating: float,
        cluster: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Assigns a persona based on multi-dimensional heuristics and cluster context.
        """
        # 1. Critical sentiment check
        if avg_rating < 2.8 and frequency >= 2:
            name = "Critical Reviewers"
        # 2. Champions: Low recency, high frequency, high monetary
        elif recency <= 60 and frequency >= 8 and monetary >= 250:
            name = "Champions (VIPs)"
        # 3. At-Risk: Long recency despite having good historical spend
        elif recency > 150 and frequency >= 4:
            name = "At-Risk Customers"
        # 4. Loyal Customers
        elif frequency >= 5 and recency <= 120:
            name = "Loyal Customers"
        # 5. Potential Loyalists: Recent buyers with moderate spend
        elif recency <= 90 and frequency >= 2:
            name = "Potential Loyalists"
        # 6. Default: Hibernating / Casuals
        else:
            name = "Hibernating / Casuals"

        playbook = PERSONA_PLAYBOOKS[name].copy()
        playbook["persona_name"] = name
        return playbook

    @classmethod
    def enrich_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enriches an RFM DataFrame with persona labels and priority levels.
        """
        personas = []
        priorities = []
        strategies = []
        channels = []
        incentives = []

        for _, row in df.iterrows():
            p = cls.assign_persona(
                recency=row.get("recency", 100),
                frequency=row.get("frequency", 2),
                monetary=row.get("monetary", 50),
                avg_rating=row.get("avg_rating", 4.0),
                cluster=row.get("cluster")
            )
            personas.append(p["persona_name"])
            priorities.append(p["priority_level"])
            strategies.append(p["marketing_strategy"])
            channels.append(p.get("preferred_channel", "Email / Retargeting Ads"))
            incentives.append(p.get("discount_incentive", "Personalized Loyalty Promo"))

        result = df.copy()
        result["persona"] = personas
        result["priority_level"] = priorities
        result["marketing_strategy"] = strategies
        result["preferred_channel"] = channels
        result["discount_incentive"] = incentives
        return result

