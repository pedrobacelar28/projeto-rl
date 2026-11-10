import numpy as np
import random
import math
import pandas as pd
import copy
from datetime import datetime, timedelta


class ActivityExecutioner():
    def __init__(self, random_obj = random.Random()):
        #GENERAL params
        self.fixed_cost = 100
        self.length = 10
        self.parallel_structure = {"HQ": ["contact_headquarters", "skip_contact"], "val": ["validate_application", "call_customer", "email_customer"]}
        #ACTIVITY params
        self.costs_dic = {"initiate_application": 0,
                    "start_standard": 10,
                    "start_priority": 5000,
                    "validate_application": 20,
                    "contact_headquarters": 3000,
                    "skip_contact": 0,
                    "email_customer": 10,
                    "call_customer": 20,
                    "calculate_offer": 400,
                    "cancel_application": 30,
                    "receive_acceptance": 10,
                    "receive_refusal": 10,
                    "stop_application": 0}
        self.times_dic = {
                    "initiate_application": 0.1,
                    "start_standard": 1,
                    "start_priority": 5,
                    "validate_application": 0.7,
                    "contact_headquarters": 6,
                    "skip_contact": 0,
                    "email_customer": 0.8,
                    "call_customer": 0.6,
                    "calculate_offer": 1,
                    "cancel_application": 0.1,
                    "receive_acceptance": 0.2,
                    "receive_refusal": 0.2,
                    "stop_application": 0}
        self.times_dic = {key: value * 86400 for key, value in self.times_dic.items()}
        #RANDOM params
        self.random_obj = random_obj

    def set_state(self, random_state):
        self.random_obj.setstate(random_state)

    def decrease_costs_contact_headquarters(self, uncertainty):
        multiplier = 1000
        self.costs_dic["contact_headquarters"] = uncertainty * multiplier + multiplier

    # CALCULATE OUTCOME
    def calc_outcome(self, current_event):
        if current_event["activity"] == "receive_acceptance":
            i = current_event["interest_rate"]
            A = current_event["amount"]
            risk_factor = (10 - current_event["quality"]) / 200
            risk_free_rate = 0.03 # around the 10 year Belgian government bond yield 16/11/2023, also accounts for inflation
            df = risk_free_rate + risk_factor
            n = self.length # number of years
            future_earnings = A * (1 + i)**n
            discount_future_earnings = future_earnings / (1 + df)**n
            exp_profit = discount_future_earnings - current_event["cum_cost"] - A - self.fixed_cost
            return exp_profit
        else: 
            return -current_event["cum_cost"] - self.fixed_cost


    # SAMPLE FUNCTIONS
    def sample_amount(self):
        mean = 30000
        std_dev = 25000
        amount_min = 3000
        amount_max = 1000000
        ok = False
        while not ok:
            sample = self.random_obj.gauss(mean, std_dev)
            sample = int(sample)
            if sample >= amount_min and sample <= amount_max:
                ok = True
        return sample


    def sample_quality(self, min_quality = 1, max_quality = 10, quality_mean = 4):
        mean = quality_mean
        std_dev = 2
        amount_min = min_quality
        amount_max = max_quality
        ok = False
        while not ok:
            sample = self.random_obj.gauss(mean, std_dev)
            sample = int(sample)
            if sample >= amount_min and sample <= amount_max:
                ok = True
        return sample


    def sample_unc_quality(self):
        mean = 3
        std_dev = 4
        amount_min = 1
        amount_max = 5
        ok = False
        while not ok:
            sample = self.random_obj.gauss(mean, std_dev)
            sample = int(sample)
            if sample >= amount_min and sample <= amount_max:
                ok = True

        self.decrease_costs_contact_headquarters(sample)
        
        return sample


    def sample_est_quality(self, quality, unc):
        mean = quality
        std_dev = unc*(1/3)
        amount_min = 0
        amount_max = 10
        ok = False
        while not ok:
            sample = self.random_obj.gauss(mean, std_dev)
            sample = int(sample)
            if sample >= amount_min and sample <= amount_max:
                ok = True
        return sample


    def sample_decrease_unc_quality(self, contact_mode, prev_event):
        if contact_mode == "email_customer":
            mean = 2
            std_dev = 0
            amount_min = 0
            amount_max = 4
            ok = False
            while not ok:
                sample = self.random_obj.gauss(mean, std_dev)
                sample = int(sample)
                if sample >= amount_min and sample <= amount_max:
                    ok = True

        if contact_mode == "call_customer":
            mean = 3
            std_dev = 0
            amount_min = 1
            amount_max = 5
            ok = False
            while not ok:
                sample = self.random_obj.gauss(mean, std_dev)
                sample = int(sample)
                if sample >= amount_min and sample <= amount_max:
                    ok = True
        sample = max(sample, 0)
        
        new_uncertainty = max(prev_event["unc_quality"] - sample, 0)

        self.decrease_costs_contact_headquarters(new_uncertainty)

        return new_uncertainty


    def sample_parallel_margin(self):
        mean = 250
        sample = mean
        return sample


    # CALCULATE OFFER
    def calculate_offer(self, prev_event, intervention_info = None):
        #ENVIRONMENT
        interest_rate = .01 #baseline
        # interest rate should be set so that still profitable, looking at discount factor, cum_cost, amount
        # this is the condition that should hold for the interest rate to be profitable
        risk_factor = (10 - prev_event["est_quality"]) / 200
        risk_free_rate = 0.03 # around the 10 year Belgian government bond yield 16/11/2023, also accounts for inflation
        df = risk_free_rate + risk_factor
    
        best_case_costs = prev_event["cum_cost"] + self.costs_dic["receive_acceptance"] + self.fixed_cost
        min_interest_rate = ((best_case_costs / prev_event["amount"] + 1)**(1 / self.length)) * (1 + df) - 1
        min_interest_rate = math.ceil(min_interest_rate * 100) / 100
        interest_rate = min_interest_rate

        if prev_event["nor"] > 0:
            interest_rate = prev_event["interest_rate"] - .01 # improve offer
        else: # first offer:
            amount = prev_event["amount"]
            quality = prev_event["est_quality"]
            elapsed_time = prev_event["elapsed_time"]

            amount_levels = {60000: 0.07, 30000: 0.08, 0: 0.09}
            elapsed_time_min = 6
            for a_key, ir in amount_levels.items():
                if amount > a_key:
                    interest_rate = ir
                    if elapsed_time < elapsed_time_min:
                        interest_rate += 0.01
                    break

            interest_rate = max(interest_rate, min_interest_rate)
            three_levels = [0.07, 0.08, 0.09]
            interest_rate = min(three_levels, key=lambda x:abs(x-interest_rate))
            if intervention_info["RCT"]:
                interest_rate = self.random_obj.choice(three_levels)

        return interest_rate, min_interest_rate, df


    def set_simulation_end_and_start(self, simulation_start, last_event):
        simulation_end = last_event["timestamp"] + timedelta(seconds=self.times_dic[last_event["activity"]]) + timedelta(days=1)
        simulation_start = simulation_end
        return simulation_start, simulation_end


    # SET EVENT TIMESTAMP
    def set_event_timestamp(self, current_activity, prev_event, env, parallel_executions, parallel_timestamps, simulation_start):
        start_timestamp_parallel_HQ, end_timestamp_parallel_HQ = parallel_timestamps["HQ"][0], parallel_timestamps["HQ"][1]
        start_timestamp_parallel_val, end_timestamp_parallel_val = parallel_timestamps["val"][0], parallel_timestamps["val"][1]
        prev_max_end_timestamp = max(end_timestamp_parallel_HQ, end_timestamp_parallel_val)
        timeout = 0
        parallel_margin = self.sample_parallel_margin()

        if prev_event is not None:
            first_parallel = ((current_activity in self.parallel_structure["HQ"] and prev_event["activity"] in self.parallel_structure["val"]) or (current_activity in self.parallel_structure["val"] and prev_event["activity"] in self.parallel_structure["HQ"]))
        else:
            first_parallel = False
        
        if (current_activity in self.parallel_structure["HQ"] or current_activity in self.parallel_structure["val"]) and (first_parallel or parallel_executions):
            if parallel_executions == False:
                # First activity that is executed after/during an execution with which it is parallel
                if current_activity in self.parallel_structure["HQ"]:
                    # If in HQ --> the parallel executions start only "after" this activity, as contact_headquarters/skip_contact needs the decision of the previous activity and thus needs to await its execution
                    if prev_event is None:
                        timestamp = simulation_start + timedelta(seconds=env.now)
                    else:
                        timestamp = prev_event["timestamp"] + timedelta(seconds=self.times_dic[prev_event["activity"]])
                    start_timestamp_parallel_HQ = timestamp
                    start_timestamp_parallel_val = timestamp + timedelta(seconds=parallel_margin)
                    end_timestamp_parallel_HQ = timestamp + timedelta(seconds=self.times_dic[current_activity])
                    end_timestamp_parallel_val = timestamp + timedelta(seconds=parallel_margin)
                    prev_max_end_timestamp = timestamp
                else:
                    # allow it to already be parallel to previous activity (previous is contact/skip)
                    timestamp = prev_event["timestamp"] + timedelta(seconds=parallel_margin)
                    start_timestamp_parallel_HQ = prev_event["timestamp"]
                    start_timestamp_parallel_val = timestamp
                    end_timestamp_parallel_HQ = prev_event["timestamp"] + timedelta(seconds=self.times_dic[prev_event["activity"]])
                    end_timestamp_parallel_val = timestamp + timedelta(seconds=self.times_dic[current_activity])
                    prev_max_end_timestamp = timestamp
                parallel_executions = True
            else:
                # Non-first parallel execution
                if current_activity in self.parallel_structure["HQ"]:
                    timestamp = end_timestamp_parallel_HQ
                    end_timestamp_parallel_HQ += timedelta(seconds=self.times_dic[current_activity])
                else:
                    timestamp = end_timestamp_parallel_val
                    end_timestamp_parallel_val += timedelta(seconds=self.times_dic[current_activity])

            duration = ((timestamp + timedelta(seconds=self.times_dic[current_activity])) - prev_max_end_timestamp).total_seconds()
            timeout = max(0, duration)

        else:
            if parallel_executions == True:
                # Back to sequential
                parallel_executions = False
                parallel_end = max(end_timestamp_parallel_HQ, end_timestamp_parallel_val)
                # Time out for the parallel executions (for which the environment did not time out yet)
                timeout = self.times_dic[current_activity]
                timestamp = parallel_end
            else:
                # Normal timestamp and timeout
                if prev_event is None:
                    timestamp = simulation_start + timedelta(seconds=env.now)
                else:
                    timestamp = prev_event["timestamp"] + timedelta(seconds=self.times_dic[prev_event["activity"]])
                timeout = self.times_dic[current_activity]
        
        parallel_timestamps = {"HQ": [start_timestamp_parallel_HQ, end_timestamp_parallel_HQ] , "val": [start_timestamp_parallel_val, end_timestamp_parallel_val]}

        return timestamp, parallel_executions, parallel_timestamps, timeout


    # SET OTHER EVENT VARIABLES
    def set_event_variables(self, current_event, prev_event, action_to_be_taken = None, intervention_info = None):
        current_activity = current_event["activity"]

        if current_activity == "initiate_application":
            current_event["amount"] = self.sample_amount()
            current_event["quality"] = self.sample_quality()
            current_event["unc_quality"] = self.sample_unc_quality()
            current_event["est_quality"] = self.sample_est_quality(current_event["quality"], current_event["unc_quality"])
            current_event["cum_cost"] = self.costs_dic[current_activity]
            current_event["interest_rate"] = np.nan
            current_event["discount_factor"] = np.nan
            current_event["outcome"] = np.nan
            current_event["noc"] = 0
            current_event["nor"] = 0
            current_event["min_interest_rate"] = np.nan
        
        elif current_activity == "call_customer" or current_activity == "email_customer":
            current_event["amount"] = prev_event["amount"]
            current_event["quality"] = prev_event["quality"]
            current_event["unc_quality"] = self.sample_decrease_unc_quality(current_activity, prev_event)
            current_event["est_quality"] = self.sample_est_quality(current_event["quality"], current_event["unc_quality"])
            current_event["cum_cost"] = prev_event["cum_cost"] + self.costs_dic[current_activity]
            current_event["interest_rate"] = prev_event["interest_rate"]
            current_event["discount_factor"] = np.nan
            current_event["outcome"] = np.nan
            current_event["noc"] = prev_event["noc"] + 1
            current_event["nor"] = prev_event["nor"]
            current_event["min_interest_rate"] = prev_event["min_interest_rate"]
        
        elif current_activity == "calculate_offer":
            current_event["amount"] = prev_event["amount"]
            current_event["quality"] = prev_event["quality"]
            current_event["unc_quality"] = prev_event["unc_quality"]
            current_event["est_quality"] = prev_event["est_quality"]
            current_event["cum_cost"] = prev_event["cum_cost"] + self.costs_dic[current_activity]
            current_event["interest_rate"], current_event["min_interest_rate"], current_event["discount_factor"] = self.calculate_offer(prev_event, intervention_info)
            current_event["outcome"] = np.nan
            current_event["noc"] = prev_event["noc"]
            current_event["nor"] = prev_event["nor"]
        
        elif current_activity == "cancel_application" or current_activity == "receive_acceptance":
            current_event["amount"] = prev_event["amount"]
            current_event["quality"] = prev_event["quality"]
            current_event["unc_quality"] = prev_event["unc_quality"]
            current_event["est_quality"] = prev_event["est_quality"]
            current_event["cum_cost"] = prev_event["cum_cost"] + self.costs_dic[current_activity]
            current_event["interest_rate"] = prev_event["interest_rate"]
            current_event["discount_factor"] = prev_event["discount_factor"]
            current_event["noc"] = prev_event["noc"]
            current_event["nor"] = prev_event["nor"]
            current_event["min_interest_rate"] = prev_event["min_interest_rate"]
            current_event["outcome"] = self.calc_outcome(current_event)

        elif current_activity == "receive_refusal":
            current_event["amount"] = prev_event["amount"]
            current_event["quality"] = prev_event["quality"]
            current_event["unc_quality"] = prev_event["unc_quality"]
            current_event["est_quality"] = prev_event["est_quality"]
            current_event["cum_cost"] = prev_event["cum_cost"] + self.costs_dic[current_activity]
            current_event["interest_rate"] = prev_event["interest_rate"]
            current_event["discount_factor"] = prev_event["discount_factor"]
            current_event["outcome"] = np.nan
            current_event["noc"] = prev_event["noc"]
            current_event["nor"] = prev_event["nor"] + 1
            current_event["min_interest_rate"] = prev_event["min_interest_rate"]
        
        else:
            current_event["amount"] = prev_event["amount"]
            current_event["quality"] = prev_event["quality"]
            current_event["unc_quality"] = prev_event["unc_quality"]
            current_event["est_quality"] = prev_event["est_quality"]
            if current_activity == "contact_headquarters":
                self.decrease_costs_contact_headquarters(prev_event["unc_quality"])
            current_event["cum_cost"] = prev_event["cum_cost"] + self.costs_dic[current_activity]
            current_event["interest_rate"] = prev_event["interest_rate"]
            current_event["discount_factor"] = prev_event["discount_factor"]
            current_event["outcome"] = np.nan
            current_event["noc"] = prev_event["noc"]
            current_event["nor"] = prev_event["nor"]
            current_event["min_interest_rate"] = prev_event["min_interest_rate"]
        
        #COUNTERFACTUAL
        if action_to_be_taken is not None and intervention_info is not None:
            for activity_index, activities in enumerate(intervention_info["activities"]):
                if current_activity in activities:
                    current_event[intervention_info["column"][activity_index]] = action_to_be_taken
                    break

        return current_event