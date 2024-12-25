import os
from roboflow import Roboflow
rf = Roboflow(api_key="1YWULeEHQvXSrJWvOr4v")
project = rf.workspace("ecommerce-clothes").project("worksite-safety-monitoring")
version = project.version(1)
dataset = version.download("yolov8")

