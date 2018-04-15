# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 06:41:53 2018

@author: HP_OWNER
"""

import tensorflow as tf

#f = open('C:\\Users\\HP_OWNER\\Desktop\\TensorFlow-ESPCN\\Tools\\test.txt', 'w')

'''new_graph = tf.Graph()
with  tf.Session(graph=new_graph) as sess:
    saver = tf.train.import_meta_graph("C:\\Users\\HP_OWNER\\Desktop\\TensorFlow-ESPCN\\checkpoint\\MC_ST.model-54500.meta", clear_devices=True)
    sess.run(tf.global_variables_initializer())
    
    
    all_vars = tf.all_variables()
    mode_zero_vars = [k for k in all_vars if 'Adam' not in k.name and 'sTransformer' in k.name]
    saver2 = tf.train.Saver(mode_zero_vars)
    saver2.save(sess, 'C:\\Users\\HP_OWNER\\Desktop\\TensorFlow-ESPCN\\checkpoint\\cool', 10)'''


new_graph = tf.Graph()
with  tf.Session(graph=new_graph) as sess:
    saver = tf.train.import_meta_graph("C:\\Users\\HP_OWNER\\Desktop\\TensorFlow-ESPCN\\checkpoint\\cool-10.meta", clear_devices=True)
    sess.run(tf.global_variables_initializer())
    
    #all_vars = tf.all_variables()
    #mode_zero_vars = [k.name for k in all_vars if 'Adam' not in k.name and 'sTransformer' in k.name]
    #print(mode_zero_vars)
    
    for n in tf.get_default_graph().as_graph_def().node:
        print(n.name)

#print(mode_zero_vars)

#for n in tf.get_default_graph().as_graph_def().node:
    #print(n.name)
    

    
#print(tf.get_default_graph().get_all_collection_keys())
#for v in tf.get_default_graph().get_collection("variables"):
    #print(v)
#for v in tf.get_default_graph().get_collection("trainable_variables"):
    #print(v)
#sess = tf.Session()
#saver.restore(sess, "C:\\Users\\HP_OWNER\\Desktop\\TensorFlow-ESPCN\\checkpoint\\MC_ST_32_3\\MC_ST.model-324000.meta")
#result = sess.run("v4:0", feed_dict={"v1:0": 12.0, "v2:0": 4.0})
#print(result)
