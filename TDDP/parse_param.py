
import re
from enum import Enum

from truthdiscovery import MatrixDataset
from truthdiscovery.algorithm import CRH
from truthdiscovery.algorithm import PriorBelief
from truthdiscovery.utils import (
    ConvergenceIterator,
    DistanceMeasures,
    filter_dict,
    FixedIterator
)

class OutputFields(Enum):
    """
    Fields available to show in results to a user
    """
    ACCURACY = "accuracy"
    BELIEF = "belief"
    BELIEF_STATS = "belief_stats"
    ITERATIONS = "iterations"
    MOST_BELIEVED = "most_believed_values"
    TIME = "time"
    TRUST = "trust"
    TRUST_STATS = "trust_stats"

class ParseParam:
    def get_param_dict(self, params_str):
        """
        Parse a multi-line string of parameters to a dictionary where keys are
        parameter names, and values are parameter values as their proper Python
        types
        """
        params = {}
        if params_str is not None:
            for line in params_str.split("\n"):
                key, value = self.algorithm_parameter(line)
                params[key] = value
        return params

    def algorithm_parameter(self, param_string):
        """
        Parse a string representation of a parameter to construct an algorithm
        object with

        :return: a pair ``(param_name, value)``
        """
        try:
            param, value = map(str.strip, param_string.split("=", maxsplit=1))
        except ValueError:
            raise ValueError(
                "parameters must be in the form 'key=value'"
            )
        # Map param name to a callable to convert string to correct type
        type_mapping = {
            "iterator": self.get_iterator,
            "priors": PriorBelief
        }
        type_convertor = type_mapping.get(param, float)
        return (param, type_convertor(value))

    def get_iterator(self, it_string, max_limit=200):
        """
        Parse an :any:`Iterator` object from a string representation
        """
        fixed_regex = re.compile(r"fixed-(?P<limit>\d+)$")
        convergence_regex = re.compile(
            r"(?P<measure>[^-]+)-convergence-(?P<threshold>[^-]+)"
            r"(-limit-(?P<limit>\d+))?$"  # optional limit
        )
        fixed_match = fixed_regex.match(it_string)
        if fixed_match:
            limit = int(fixed_match.group("limit"))
            if limit > max_limit:
                raise ValueError(
                    "Cannot perform more than {} iterations".format(max_limit)
                )
            return FixedIterator(limit=limit)

        convergence_match = convergence_regex.match(it_string)
        if convergence_match:
            measure_str = convergence_match.group("measure")
            try:
                measure = DistanceMeasures(measure_str)
            except ValueError:
                raise ValueError(
                    "invalid distance measure '{}'".format(measure_str)
                )
            threshold = float(convergence_match.group("threshold"))
            limit = max_limit
            if convergence_match.group("limit") is not None:
                limit = int(convergence_match.group("limit"))
                if limit > max_limit:
                    raise ValueError(
                        "Upper iteration limit cannot exceed {}"
                        .format(max_limit)
                    )
            return ConvergenceIterator(measure, threshold, limit)

        raise ValueError(
            "invalid iterator specification '{}'".format(it_string)
        )

    def get_output_obj(self, results, output_fields=None, sup_data=None):
        """
        TD 算法 run 完之后的结果传入到这里来读取详细信息。

        Format a :any:`Result` class as a dictionary to present as output to
        the user. If ``output_fields`` is None, include all available fields
        """
        output_fields = output_fields or list(OutputFields)
        out = {}

        for field in output_fields:
            if field == OutputFields.TIME:
                out[field.value] = results.time_taken

            if field == OutputFields.ITERATIONS:
                out[field.value] = results.iterations

            if field == OutputFields.TRUST:
                out[field.value] = results.trust

            if field == OutputFields.BELIEF:
                out[field.value] = results.belief

            if field == OutputFields.TRUST_STATS:
                mean, stddev = results.get_trust_stats()
                out[field.value] = {"mean": mean, "stddev": stddev}

            if field == OutputFields.BELIEF_STATS:
                belief_stats = results.get_belief_stats()
                out[field.value] = {
                    var: {"mean": mean, "stddev": stddev}
                    for var, (mean, stddev) in belief_stats.items()
                }

            if sup_data is not None and field == OutputFields.ACCURACY:
                try:
                    acc = sup_data.get_accuracy(results)
                except ValueError:
                    acc = None
                out[field.value] = acc

            if field == OutputFields.MOST_BELIEVED:
                most_bel = {}
                for var in results.belief:
                    most_bel[var] = sorted(
                        results.get_most_believed_values(var)
                    )
                out[field.value] = most_bel

        return out
