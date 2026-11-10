import numpy as np
import random
import math
import pandas as pd
import copy
from pm4py.objects.petri_net import semantics
from pm4py.objects.petri_net.obj import PetriNet
from pm4py.objects.petri_net.utils.petri_utils import get_transition_by_name

class ExtraFlowConditioner():
    def __init__(self, random_obj = random.Random()):
        #GENERAL params
        self.random_obj = random_obj

    def filter_enabled_trans(self, net, marking, control_flow_enabled_trans, trace, policies_info, intervention_info, action_to_be_taken = None, ignore_intervention_policy = False):
        prev_event = trace[-1] if len(trace) > 0 else None
        marking = str(marking)
        policies = {}

        # STATING ALL ADDITIONAL CF CONDITIONS
        if prev_event is not None:
            # Priority policy
            policies["choose_procedure"] = (prev_event["amount"] > policies_info["choose_procedure"]["amount"] and prev_event["est_quality"] >= policies_info["choose_procedure"]["est_quality"])

            # Contact HQ policy
            policies["time_contact_HQ"] = {"fast": True, "slow": False, "real": (prev_event["noc"] < 2 and prev_event["unc_quality"] == 0 and prev_event["amount"] > 10000 and prev_event["est_quality"] >= policies_info["min_quality"])}

            # Cancel policy
            policies["cancel_application"] = {"quality": (prev_event["unc_quality"] == 0 and prev_event["est_quality"] < policies_info["min_quality"] and prev_event["noc"] >= policies_info["max_noc"]),
                                            "noc": (prev_event["noc"] >= policies_info["max_noc"] and (prev_event["unc_quality"] > 0)),
                                            "nor": (prev_event["nor"] >= policies_info["max_nor"]),
                                            "no_contact_HQ_bank_policy": (prev_event["noc"] >= policies_info["max_noc"] and not policies["time_contact_HQ"][policies_info["time_contact_HQ"]]),
                                            "interest_rate": (prev_event["interest_rate"] < prev_event["min_interest_rate"]),
                                            "skip_contact": (any(event.get("activity") == "skip_contact" for event in trace))}

            # Contact customer policy
            policies["call_customer"] = (prev_event["amount"] > policies_info["min_amount_contact_cust"])
            
            # Calculate offer policy
            policies["calculate_offer"] = (prev_event["unc_quality"] == 0 and prev_event["noc"] >= policies_info["max_noc"])

            # Skip contact policy
            policies["skip_contact"] = (prev_event["noc"] >= policies_info["max_noc"]) and (prev_event["activity"] == "validate_application")
        
        if intervention_info["RCT"] or ignore_intervention_policy:
            policies_to_ignore = intervention_info["name"]
        else:
            policies_to_ignore = []

        if marking == "['source:1']":
            all_enabled_trans = control_flow_enabled_trans
        
        elif marking == "['p_proc:1']":
            if policies["choose_procedure"]:
                all_enabled_trans = [act for act in control_flow_enabled_trans if act not in [get_transition_by_name(net, "start_standard")]]
            else:
                all_enabled_trans = [act for act in control_flow_enabled_trans if act not in [get_transition_by_name(net, "start_priority")]]
            if "choose_procedure" in policies_to_ignore:
                all_enabled_trans = control_flow_enabled_trans

        elif marking == "['p_stand_cont:1', 'p_stand_val:1']":
            # if cancel, should delete contact hq
            # if contact possible, should delete skip contact
            # in other cases, should delete both
            # if cancel_pol_conds["quality"] or cancel_pol_conds["noc"]:
            if policies["cancel_application"]["quality"] or policies["cancel_application"]["noc"] or policies["cancel_application"]["no_contact_HQ_bank_policy"]:
                all_enabled_trans = [act for act in control_flow_enabled_trans if act not in [get_transition_by_name(net, "contact_headquarters")]]

            elif policies["time_contact_HQ"][policies_info["time_contact_HQ"]]:
                if get_transition_by_name(net, "contact_headquarters") in control_flow_enabled_trans:
                    all_enabled_trans = [get_transition_by_name(net, "contact_headquarters")]
                else:
                    all_enabled_trans = [act for act in control_flow_enabled_trans if act not in [get_transition_by_name(net, "skip_contact")]]
            else:
                all_enabled_trans = [act for act in control_flow_enabled_trans if act not in [get_transition_by_name(net, "contact_headquarters"), get_transition_by_name(net, "skip_contact")]]
                
            # IGNORE POLICY CONTACT HQ
            if "time_contact_HQ" in policies_to_ignore:
                all_enabled_trans = control_flow_enabled_trans
                if not policies["skip_contact"]:
                    all_enabled_trans = [act for act in all_enabled_trans if act not in [get_transition_by_name(net, "skip_contact")]]

            
        elif marking == "['p_stand_cont:1', 'p_val_ghost_calc:1']":
            if policies["cancel_application"]["quality"] or policies["cancel_application"]["noc"] or policies["cancel_application"]["no_contact_HQ_bank_policy"]:
                all_enabled_trans = [act for act in control_flow_enabled_trans if act not in [get_transition_by_name(net, "contact_headquarters"), get_transition_by_name(net, "call_customer"), get_transition_by_name(net, "email_customer")]]
                
                if "time_contact_HQ" in policies_to_ignore:
                    # NOTE: Call and email are not allowed in this case, as should be cancelled afterwards
                    all_enabled_trans = [act for act in control_flow_enabled_trans if act not in [get_transition_by_name(net, "call_customer"), get_transition_by_name(net, "email_customer")]]
                    if not policies["skip_contact"]:
                        all_enabled_trans = [act for act in all_enabled_trans if act not in [get_transition_by_name(net, "skip_contact")]]

            elif policies["time_contact_HQ"][policies_info["time_contact_HQ"]]:
                if get_transition_by_name(net, "contact_headquarters") in control_flow_enabled_trans:
                    all_enabled_trans = [get_transition_by_name(net, "contact_headquarters")]
                else:
                    all_enabled_trans = [act for act in control_flow_enabled_trans if act not in [get_transition_by_name(net, "skip_contact")]]
                if "time_contact_HQ" in policies_to_ignore:
                    all_enabled_trans = control_flow_enabled_trans
                    if not policies["skip_contact"]:
                        all_enabled_trans = [act for act in control_flow_enabled_trans if act not in [get_transition_by_name(net, "skip_contact")]]

                if policies["calculate_offer"]:
                    all_enabled_trans = [act for act in all_enabled_trans if act not in [get_transition_by_name(net, "call_customer"), get_transition_by_name(net, "email_customer")]]

                if policies["call_customer"]:
                    all_enabled_trans = [act for act in all_enabled_trans if act not in [get_transition_by_name(net, "email_customer")]]

                else:
                    all_enabled_trans = [act for act in all_enabled_trans if act not in [get_transition_by_name(net, "call_customer")]]
                
            else:
                all_enabled_trans = [act for act in control_flow_enabled_trans if act not in [get_transition_by_name(net, "skip_contact"), get_transition_by_name(net, "contact_headquarters")]]
                if "time_contact_HQ" in policies_to_ignore:
                    all_enabled_trans = control_flow_enabled_trans
                    if not policies["skip_contact"]:
                        all_enabled_trans = [act for act in control_flow_enabled_trans if act not in [get_transition_by_name(net, "skip_contact")]]
                
                if policies["call_customer"]:
                    all_enabled_trans = [act for act in all_enabled_trans if act not in [get_transition_by_name(net, "email_customer")]]

                else:
                    all_enabled_trans = [act for act in all_enabled_trans if act not in [get_transition_by_name(net, "call_customer")]]
            

        elif marking == "['p_cont_ghost_calc:1', 'p_stand_val:1']":
            all_enabled_trans = control_flow_enabled_trans
            
        elif marking == "['p_cont_ghost_calc:1', 'p_val_ghost_calc:1']":
            if policies["cancel_application"]["quality"] or policies["cancel_application"]["noc"]:
                all_enabled_trans = [act for act in control_flow_enabled_trans if act not in [get_transition_by_name(net, "call_customer"), get_transition_by_name(net, "email_customer")]]
            elif not policies["calculate_offer"]:
                all_enabled_trans = [act for act in control_flow_enabled_trans if act not in [get_transition_by_name(net, "ghost_calc")]]
                if policies["call_customer"]:
                    all_enabled_trans = [act for act in all_enabled_trans if act not in [get_transition_by_name(net, "email_customer")]]
                else:
                    all_enabled_trans = [act for act in all_enabled_trans if act not in [get_transition_by_name(net, "call_customer")]]
            else:
                all_enabled_trans = [act for act in control_flow_enabled_trans if act not in [get_transition_by_name(net, "call_customer"), get_transition_by_name(net, "email_customer")]]

        elif marking == "['p_ghost_calc:1']":
            if prev_event["activity"] == "start_priority":
                all_enabled_trans = [act for act in control_flow_enabled_trans if act not in [get_transition_by_name(net, "ghost_canc_before")]]

            elif policies["cancel_application"]["quality"] or policies["cancel_application"]["noc"] or policies["cancel_application"]["nor"] or policies["cancel_application"]["skip_contact"]:
                all_enabled_trans = [act for act in control_flow_enabled_trans if act not in [get_transition_by_name(net, "calculate_offer")]]

            else:
                all_enabled_trans = [act for act in control_flow_enabled_trans if act not in [get_transition_by_name(net, "ghost_canc_before")]]

        elif marking == "['p_ghost_canc:1']":
            all_enabled_trans = control_flow_enabled_trans
            
        elif marking == "['p_calc_acc:1']":
            if policies["cancel_application"]["interest_rate"]:
                all_enabled_trans = [act for act in control_flow_enabled_trans if act not in [get_transition_by_name(net, "receive_acceptance"), get_transition_by_name(net, "receive_refusal")]]
            else:
                all_enabled_trans = [act for act in control_flow_enabled_trans if act not in [get_transition_by_name(net, "ghost_canc_after")]]

                activity_to_delete = self.get_customer_decision_logic(net, prev_event)
                all_enabled_trans = [act for act in all_enabled_trans if act not in [activity_to_delete]]
        

        elif marking == "['p_ghost_calc:1']":
            all_enabled_trans = control_flow_enabled_trans
            
        elif marking == "['p_calc_acc:1']":
            all_enabled_trans = control_flow_enabled_trans
            
        #COUNTERFACTUAL
        if action_to_be_taken is not None and action_to_be_taken != "do_nothing":
            all_enabled_trans = [get_transition_by_name(net, action_to_be_taken)]

        return all_enabled_trans



    def get_customer_decision_logic(self, net, prev_event):
        def calculate_acceptance_probability(amount, interest_rate, elapsed_time):
            # Continuous function to calculate probability adjustments based on amount and interest rate
            def prob_adjustment(amount, interest_rate):
                # Adjust based on amount
                max_amount = 80000
                min_amount = 1000
                if amount > max_amount:
                    a_factor = 0.9999
                else:
                    a_factor = (amount - min_amount) / (max_amount - min_amount)
                
                # # Adjust based on interest rate
                max_interest = 0.091
                min_interest = 0.069
                i_factor = (interest_rate - min_interest) / (max_interest - min_interest)

                # Higher amounts with higher interest rates should decrease probability
                combined_factor = a_factor * i_factor
                # Sacle combined factor
                combined_factor = -2 * combined_factor + 1

                return combined_factor

            # Continuous function to adjust probability based on elapsed time
            def time_adjustment(elapsed_time):
                max_time = 13
                min_time = 4.9
                time_factor = (elapsed_time - min_time) / (max_time - min_time)
                # Scale factor to
                time_factor = -2 * time_factor + 1
                
                # Higher elapsed times decrease probability
                return time_factor

            # Initialize baseline probability (0.5 as a neutral starting point)
            prob = 0.5

            # Apply the adjustments based on amount and interest rate
            prob_adjust = prob_adjustment(amount, interest_rate) * 0.25
            prob += prob_adjust

            # Apply adjustments based on elapsed time
            prob_time = time_adjustment(elapsed_time) * 0.25
            prob += prob_time

            # Now, map the final probability to the appropriate range
            if prob < 0.5:
                final_prob = prob * 0.6  
            else:
                # Map to the range for favorable conditions
                final_prob = 0.7 + (prob - 0.5) * 0.6  

            # Max and Min boundaries
            max_interest_rate = 0.12
            max_elapsed_time = 12
            if interest_rate >= max_interest_rate or elapsed_time >= max_elapsed_time:
                final_prob = 0  # Immediate rejection if interest or time exceeds the maximum

            # Ensure final probability is within 0.0001 to 0.9999
            final_prob = max(0.0001, final_prob)
            final_prob = min(0.9999, final_prob)

            return final_prob

        accept_prob = calculate_acceptance_probability(prev_event["amount"], prev_event["interest_rate"], prev_event["elapsed_time"])
        activity_to_delete = self.random_obj.choices([get_transition_by_name(net, "receive_refusal"), get_transition_by_name(net, "receive_acceptance")], weights=[accept_prob, 1 - accept_prob], k=1)[0]
        return activity_to_delete