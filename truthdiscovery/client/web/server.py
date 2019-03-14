from io import BytesIO
import json

from flask import Flask, render_template, request, abort, Response

from truthdiscovery.client.base import BaseClient
from truthdiscovery.input import MatrixDataset


class route:
    """
    Class to be used a decorator instead of ``app.route`` which allows methods
    to be decorated.

    ``app.route`` cannot be used for methods since it sees an *unbound*
    function, whereas we wish for it to register as a method bound to a
    particular instance of the class.

    As such we keep a record of the parameters passed to the decorator and the
    method names, and delay actually adding the routes in flask until an
    instance has been created and passed to ``add_routes``
    """
    # Note: this list is shared amongst all ``route`` instances
    routes = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, func):
        route.routes.append((self.args, self.kwargs, func.__name__))
        return func

    @classmethod
    def add_routes(cls, flask_app, class_instance):
        for args, kwargs, func_name in route.routes:
            try:
                meth = getattr(class_instance, func_name)
            except AttributeError:
                continue
            flask_app.add_url_rule(
                *args,
                view_func=meth,
                **kwargs
            )


class WebClient(BaseClient):
    def get_algorithm_object(self, label, params_str):
        """
        :raises ValueError: if label or parameters are invalid
        """
        cls = self.algorithm_cls(label)
        params = {}
        if params_str is not None:
            for line in params_str.split("\n"):
                key, value = self.algorithm_parameter(line)
                params[key] = value

        try:
            return cls(**params)
        except TypeError:
            raise ValueError

    @route("/")
    def home_page(self):
        # Map algorithm labels to display name
        labels = {label: cls.__name__
                  for label, cls in self.ALG_LABEL_MAPPING.items()}
        return render_template(
            "index.html",
            data_json=json.dumps({"algorithm_labels": labels})
        )

    @route("/run/", methods=["GET"])
    def run(self):
        try:
            alg_label = request.args["algorithm"]
            matrix_csv = request.args["matrix"]
        except KeyError:
            return abort(400)

        matrix_csv = matrix_csv.replace("_", "")

        params = request.args.get("parameters")
        alg = self.get_algorithm_object(alg_label, params)
        dataset = MatrixDataset.from_csv(BytesIO(matrix_csv.encode()))
        results = alg.run(dataset)
        output = self.get_output_obj(results)
        return Response(json.dumps(output), mimetype="application/json")


def get_flask_app():
    client = WebClient()
    app = Flask(__name__)
    route.add_routes(app, client)
    return app


if __name__ == "__main__":
    get_flask_app().run(debug=True)
