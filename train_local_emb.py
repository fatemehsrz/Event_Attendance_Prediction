
import time 
from gensim.models import Doc2Vec
from gensim.models.doc2vec  import TaggedLineDocument




if __name__ == '__main__':

	

	#doc2vec parameters
	embedding_size = 128
	window_size = 8
	min_count = 1
	sampling_threshold = 10e-6
	negative_size = 5
	train_epoch = 5
	dm = 1 #0 = PV-DBOW; 1 = PV-DM

	worker_count = 2 #number of parallel processes


	egoList = [0, 107, 348, 414, 686, 698, 1684, 1912, 3437, 3980]

	#input corpus
	train_corpus = "ego_local.walks"

	#output model
	saved_path1 = "doc2vecModel.bin"
	saved_path2 = "ego_local256.emb"

	#train doc2vec 
	print('giving label to egos ...')


	paragraphs = TaggedLineDocument(train_corpus)

        start_time = time.time()

	print('Training ...')
	model = Doc2Vec(paragraphs, size=embedding_size, window=window_size, min_count=min_count, sample=sampling_threshold, 
		          workers=worker_count, hs=0, dm=dm, negative=negative_size, dbow_words=1, dm_concat=1, iter=train_epoch)
		         

	#save model

	########################### save vectors #####################################
	    
	f1= open(saved_path2, 'w') 
	  
	for i in range(len(egoList)):
	    
	    f1.write(str(egoList[i]))
	    
	    print('ego:', i, ':',model.docvecs[i])
	    
	    print('--------------------------------------')
	    
	    for j in model.docvecs[i]:
		
		f1.write(' '+str(j))
	    
	    f1.write('\n')  
	       

	f1.close()  
	 
	    
	model.save(saved_path1)

	print('vector_file has been saved ...')

	print('execution time ',"--- %s seconds ---" % (time.time() - start_time))
