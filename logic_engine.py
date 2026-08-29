class KnowledgeBase:
    """
    Practical 04 - Part 1: Declarative Knowledge Base.

    Stores Facts (current percepts) and Rules (Horn Clauses) and can
    run a Data-Driven Forward Chaining inference pass over them.
    """

    def __init__(self):
        self.facts = set()   # unique string facts
        self.rules = []      # list of (premise_list, conclusion_string) tuples

    def tell_fact(self, fact_string):
        self.facts.add(fact_string)

    def tell_rule(self, premise_list, conclusion_string):
        self.rules.append((premise_list, conclusion_string))

    def clear_facts(self):
        self.facts.clear()

    def forward_chain(self):
        """
        Part 2 - Step 2.1: Data-Driven Forward Chaining.
        Keeps sweeping the rule set until a full pass adds no new facts.
        """
        new_facts_added = True

        while new_facts_added:
            new_facts_added = False

            for premises, conclusion in self.rules:
                if conclusion not in self.facts:
                    # Modus Ponens check: all premises already known?
                    if all(premise in self.facts for premise in premises):
                        self.facts.add(conclusion)
                        new_facts_added = True
