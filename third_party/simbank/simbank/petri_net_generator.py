from pm4py.objects.petri_net.obj import PetriNet
from pm4py.objects.petri_net.utils import petri_utils as utils
from pm4py.objects.petri_net.obj import Marking

#CREATE PETRI NET
nodes_to_ignore = []
decision_nodes = []
decorations = {}
def create_PN(net):
    source = PetriNet.Place("source")
    sink = PetriNet.Place("sink")
    net.places.add(source)
    net.places.add(sink)

    t_init = PetriNet.Transition("initiate_application", "initiate_application")
    t_stan = PetriNet.Transition("start_standard", "start_standard")
    t_prior = PetriNet.Transition("start_priority", "start_priority")
    t_skip = PetriNet.Transition("skip_contact", "skip_contact")
    t_cont = PetriNet.Transition("contact_headquarters", "contact_headquarters")
    t_val = PetriNet.Transition("validate_application", "validate_application")
    t_call = PetriNet.Transition("call_customer", "call_customer")
    t_email = PetriNet.Transition("email_customer", "email_customer")
    t_calc = PetriNet.Transition("calculate_offer", "calculate_offer")
    t_canc = PetriNet.Transition("cancel_application", "cancel_application")
    t_acc = PetriNet.Transition("receive_acceptance", "receive_acceptance")
    t_ref = PetriNet.Transition("receive_refusal", "receive_refusal")
    t_ghost_calc = PetriNet.Transition("ghost_calc", "ghost_calc")
    t_ghost_canc_before = PetriNet.Transition("ghost_canc_before", "ghost_canc_before")
    t_ghost_canc_after = PetriNet.Transition("ghost_canc_after", "ghost_canc_after")
    t_list = [t_init, t_stan, t_prior, t_skip, t_cont, t_val, t_call, t_email, t_calc, t_canc, t_acc, t_ref, t_ghost_calc, t_ghost_canc_before, t_ghost_canc_after]

    p_proc = PetriNet.Place("p_proc")
    p_stand_cont = PetriNet.Place("p_stand_cont")
    p_stand_val = PetriNet.Place("p_stand_val")
    p_cont_ghost_calc = PetriNet.Place("p_cont_ghost_calc")
    p_val_ghost_calc = PetriNet.Place("p_val_ghost_calc")
    p_ghost_calc = PetriNet.Place("p_ghost_calc")
    p_ghost_canc = PetriNet.Place("p_ghost_canc")
    p_calc_acc = PetriNet.Place("p_calc_acc")
    p_list = [p_proc, p_stand_cont, p_stand_val, p_cont_ghost_calc, p_val_ghost_calc, p_ghost_calc, p_ghost_canc, p_calc_acc]

    for p in p_list:
            net.places.add(p)

    for t in t_list:
        net.transitions.add(t)

    utils.add_arc_from_to(source, t_init, net)
    utils.add_arc_from_to(t_init, p_proc, net)
    utils.add_arc_from_to(p_proc, t_stan, net)
    utils.add_arc_from_to(p_proc, t_prior, net)

    utils.add_arc_from_to(t_stan, p_stand_cont, net)
    utils.add_arc_from_to(p_stand_cont, t_cont, net)
    utils.add_arc_from_to(t_cont, p_cont_ghost_calc, net)
    utils.add_arc_from_to(p_stand_cont, t_skip, net)
    utils.add_arc_from_to(t_skip, p_cont_ghost_calc, net)
    utils.add_arc_from_to(p_cont_ghost_calc, t_ghost_calc, net)

    utils.add_arc_from_to(t_stan, p_stand_val, net)
    utils.add_arc_from_to(p_stand_val, t_val, net)
    utils.add_arc_from_to(t_val, p_val_ghost_calc, net)
    utils.add_arc_from_to(p_val_ghost_calc, t_call, net)
    utils.add_arc_from_to(p_val_ghost_calc, t_email, net)
    utils.add_arc_from_to(t_call, p_stand_val, net)
    utils.add_arc_from_to(t_email, p_stand_val, net)
    utils.add_arc_from_to(p_val_ghost_calc, t_ghost_calc, net)
  
    utils.add_arc_from_to(t_ghost_calc, p_ghost_calc, net)
    utils.add_arc_from_to(p_ghost_calc, t_calc, net)
    utils.add_arc_from_to(t_prior, p_ghost_calc, net)
    
    utils.add_arc_from_to(p_ghost_calc, t_ghost_canc_before, net)
    utils.add_arc_from_to(t_ghost_canc_before, p_ghost_canc, net)

    utils.add_arc_from_to(t_calc, p_calc_acc, net)

    utils.add_arc_from_to(p_calc_acc, t_ghost_canc_after, net)
    utils.add_arc_from_to(t_ghost_canc_after, p_ghost_canc, net)
  
    utils.add_arc_from_to(p_ghost_canc, t_canc, net)
    utils.add_arc_from_to(t_canc, sink, net)

    utils.add_arc_from_to(p_calc_acc, t_acc, net)
    utils.add_arc_from_to(t_acc, sink, net)

    utils.add_arc_from_to(p_calc_acc, t_ref, net)
    utils.add_arc_from_to(t_ref, p_ghost_calc, net)

    net.initial_marking = Marking()
    net.initial_marking[source] = 1
    net.final_marking = Marking()
    net.final_marking[sink] = 2

    nodes_to_ignore = [t_ghost_calc, t_ghost_canc_after, t_ghost_canc_before]
    for node in nodes_to_ignore:
        decorations[node] = {"color": "#E5E5E5"}

#RUN
def generate_petri_net():
    petri_net =  PetriNet("Petri_net")
    create_PN(petri_net)
    # vizualize_net(petri_net)
    return petri_net