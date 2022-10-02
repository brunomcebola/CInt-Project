def detectSamples(data,gain):
    detectedSamplesList = []
    average = data.mean()
    std = data.std()
    
    listOfIndex = []
    i = 0

    for value in data:
        if (value < (average - std*gain)) or (value > (average + std*gain)):
            detectedSamplesList.append(value)
            listOfIndex.append(i)
        i += 1    
    return  listOfIndex,detectedSamplesList

def removeLine(data, indexes):
    return data.drop(indexes)
    