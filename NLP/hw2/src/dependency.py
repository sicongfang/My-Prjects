def shift(stack, buff,  dgraph):
    # TODO
    #raise NotImplementedError
    if buff ==[]:
        pass
    temp =  buff.pop()
    stack.append(temp)

def left_arc(stack, buff, dgraph):
    # TODO
    #raise NotImplementedError
    dgraph.append((stack[-2],stack[-1]))
    stack.pop(-2)
    #return stack, buff, dgraph

def right_arc(stack, buff, dgraph):
    # TODO
    #raise NotImplementedError
    dgraph.append((stack[-1],stack[-2]))
    stack.pop(-1)
    #return stack, buff, dgraph

def oracle_std(stack, buff, dgraph, gold_arcs):
    # TODO
    #raise NotImplementedError()
    
    if len(stack) <= 1:
        return 'shift'
    else:
        temp1 = len([(x, y) for x, y in gold_arcs if y == stack[-1]])
        temp2 = len([(x, y) for x, y in dgraph if y == stack[-1]])
        if (stack[-2],stack[-1]) in gold_arcs:
            return 'left_arc'
        elif (stack[-1],stack[-2]) in gold_arcs and temp1 == temp2:
            return 'right_arc'
        else:
            return 'shift'


def make_transitions(buff, oracle, gold_arcs=None):
    stack = []
    dgraph = []
    configurations = []
    while (len(buff) > 0 or len(stack) > 1):
        choice = oracle(stack, buff, dgraph, gold_arcs)
        # Makes a copy. Else configuration has a reference to buff and stack.
        config_buff = list(buff)
        config_stack = list(stack)
        configurations.append([config_stack,config_buff,choice])
        if choice == 'shift':	shift(stack, buff, dgraph)
        elif choice == 'left_arc': left_arc(stack, buff, dgraph)
        elif choice == 'right_arc': right_arc(stack, buff, dgraph)
        else: return None
    return dgraph,configurations
