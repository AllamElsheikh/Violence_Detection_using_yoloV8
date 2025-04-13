from roboflow import Roboflow
rf = Roboflow(api_key="5NzzOj4ngPyotxgJWGUu")
project = rf.workspace("project-hw9bc").project("smart-schools-using-ai-and-iot3")
dataset = project.version(1).download("yolov8")