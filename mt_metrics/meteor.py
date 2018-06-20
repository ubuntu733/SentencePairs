from itertools import groupby


class Meteor(object):
    """
        Meteor evaluation metric, as defined in https://www.cs.cmu.edu/~alavie/papers/BanerjeeLavie2005-final.pdf
    """

    @staticmethod
    def evaluate(s1, s2, ngrams=1):
        """
            Evaluates the closeness of s1 to s2
        """

        s1_grams = [' '.join(s1[i:i+ngrams]) for i in range(len(s1)-ngrams+1)]
        s2_grams = [' '.join(s2[i:i+ngrams]) for i in range(len(s2)-ngrams+1)]

        match_count = len([g for g in s1_grams if g in s2_grams])

        precision = match_count / len(s2)
        recall = match_count / len(s1)

        f_mean = (10 * precision * recall) / (recall + 9 * precision + 1e-6)

        matching_indexes = []
        grams_dict = { g:0 for g in s1_grams }

        for g in s1_grams:
            try:
                matching_indexes.append(s2_grams.index(g, grams_dict[g]))
                grams_dict[g] = matching_indexes[-1] + 1
            except ValueError:
                matching_indexes.append(-1)


        penalty = 0.5 * (
            len([
                1
                for k, g in groupby(
                                enumerate(matching_indexes),
                                (lambda t:t[0]-t[1])
                            )
            ]) / (match_count + 1e-6)) ** 3

        return f_mean * (1 - penalty)
