# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 15:20:26 2015
http://isbullsh.it/2012/06/Rest-api-in-python/
@author: erhe01
"""
import urllib2, json


mg_server_url = 'http://localhost:8888/runmorphablegraphs'
input_dict = {
  "session" : "session",
  "startPose" : {
    "position" : [ 530.110290527344, 268.851318359375, 0.0 ],
    "orientation" : [ -1.40170064921927E-14, -2.11894517055238E-6, 91.4786896560991 ]
  },
  "elementaryActions" : [ {
    "action" : "walk",
    "constraints" : [ {
      "joint" : "Hips",
      "trajectoryConstraints" : [ {
        "position" : [ 530.110290527344, 268.851318359375, None ],
        "orientation" : [ None, None, None ]
      }, {
        "position" : [ 355.0, 255.0, None ],
        "orientation" : [ None, None, None ]
      }, {
        "position" : [ 195.0, 255.0, None ],
        "orientation" : [ None, None, None ]
      }, {
        "position" : [ 147.011029052734, 206.885131835938, None ],
        "orientation" : [ None, None, None ]
      }, {
        "position" : [ 60.0, 200.0, None ],
        "orientation" : [ None, None, None ]
      }, {
        "position" : [ 30.0, 200.0, None ],
        "orientation" : [ None, None, None ]
      } ]
    } ]
  }, {
    "action" : "pickRight",
    "constraints" : [ {
      "joint" : "RightHand",
      "keyframeConstraints" : [ {
        "position" : [ -30.0000009536743, 200.0, 130.0 ],
        "orientation" : [ None, None, None ],
        "semanticAnnotation" : {
          "start_contact" : True
        }
      } ]
    } ],
    "keyframeAnnotations" : [ {
      "keyframe" : "start_contact",
      "annotations" : [ {
        "event" : "attach",
        "parameters" : {
          "joint" : "RightHand",
          "target" : "part_8556319"
        }
      } ]
    } ]
  }, {
    "action" : "carryRight",
    "constraints" : [ {
      "joint" : "Hips",
      "trajectoryConstraints" : [ {
        "position" : [ 30.0000000000001, 200.0, None ],
        "orientation" : [ None, None, None ]
      }, {
        "position" : [ 35.0, 215.0, None ],
        "orientation" : [ None, None, None ]
      }, {
        "position" : [ 105.0, 285.0, None ],
        "orientation" : [ None, None, None ]
      }, {
        "position" : [ 108.39608, 436.52072, None ],
        "orientation" : [ None, None, None ]
      }, {
        "position" : [ 161.5512, 462.8008, None ],
        "orientation" : [ None, None, None ]
      }, {
        "position" : [ 191.5512, 462.8008, None ],
        "orientation" : [ None, None, None ]
      } ]
    } ]
  }, {
    "action" : "placeRight",
    "constraints" : [ {
      "joint" : "RightHand",
      "keyframeConstraints" : [ {
        "position" : [ 251.5512, 462.8008, 129.296196185303 ],
        "orientation" : [ None, None, None ],
        "semanticAnnotation" : {
          "end_contact" : True
        }
      } ]
    } ],
    "keyframeAnnotations" : [ {
      "keyframe" : "end_contact",
      "annotations" : [ {
        "event" : "detach",
        "parameters" : {
          "joint" : "RightHand",
          "target" : "part_8556319"
        }
      } ]
    } ]
  } ]
}

data = json.dumps(input_dict)
request = urllib2.Request(mg_server_url, data)
print "send message and wait for answer..."
handler = urllib2.urlopen(request)
print handler.read()
