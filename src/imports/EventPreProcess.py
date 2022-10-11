import cv2
import numpy as np
import math

class EventPreProcess:

    #ConcatenateEvents static method
    @staticmethod
    def ConcatenateEvents(event_list, time_steps, im_height=260, im_width=346):
        #Group Events into 3 channel images at the desired time_steps as described in "Unsupervised Learning of Dense Optical Flow, Depth and Egomotion from Sparse Event Data" 
        #Channel 1: count of positive events at each pixel position
        #Channel 2: count of negative events at each pixel position
        #Channel 3: normalized timestamp of events generated at pixel position
        event_images_list = []

        event_iterator = 0
        time_step = time_steps[-1] - time_steps[-2] 
        for ts in time_steps:
            print("Percentage finished: " + str(event_iterator * 100 / len(event_list)))
            event_image = np.zeros((im_height, im_width, 3), dtype=np.float32)
            counter_matrix = np.zeros((im_height, im_width))
            timestamp_matrix = np.zeros((im_height, im_width))
            
            for i in range(event_iterator, len(event_list)):
                event_iterator = i + 1
                event = np.copy(event_list[i])
                if event[2] > ts:
                    event_image[:, :, 2] = np.zeros((im_height, im_width))
                    event_image[:, :, 2] = np.divide(timestamp_matrix, counter_matrix, where=counter_matrix!=0)

                    event_images_list.append(np.copy(event_image)) 
                    break
                else:
                    if event[3] > 0:
                        event_image[int(event[1]), int(event[0]), 0] = event_image[int(event[1]), int(event[0]), 0] + 1.0
                    else:
                        event_image[int(event[1]), int(event[0]), 1] = event_image[int(event[1]), int(event[0]), 1] + 1.0
                    
                    counter_matrix[int(event[1]), int(event[0])] = counter_matrix[int(event[1]), int(event[0])] + 1.0                

                    timestamp_matrix[int(event[1]), int(event[0])] = (np.abs(ts - event[2] - time_step)/time_step) + timestamp_matrix[int(event[1]), int(event[0])]


        return event_images_list, event_list[event_iterator:len(event_list)]

    
    #FrameGeneration static method
    @staticmethod
    def generateFrames(event_list, time_steps, im_height=260, im_width=346, im_channel=1, time_window=None):
        #One Channel
        event_images_list = []

        event_iterator = 0
        for ts in time_steps:
            #print("Percentage finished: " + str(event_iterator * 100 / len(event_list)))
            event_image = np.zeros((im_height, im_width, im_channel), dtype=np.float32)
            
            for i in range(event_iterator, len(event_list)):
                event_iterator = i + 1
                event = np.copy(event_list[i]) #[x, y, ts, polarity]
                if (time_window==None) or (event[2] > ts - time_window):
                    if event[2] > ts:
                        event_image[201, 154, 0] = 0
                        event_images_list.append(np.copy(event_image)) 
                        break
                    else:
                        if event[3] > 0:
                            event_image[int(event[1]), int(event[0]), 0] = event_image[int(event[1]), int(event[0]), 0] + 1.0
                        else:
                            event_image[int(event[1]), int(event[0]), 0] = event_image[int(event[1]), int(event[0]), 0] - 1.0             

        return event_images_list, event_list[event_iterator:len(event_list)]
    
    @staticmethod
    def generateFrames_temporalROI(event_list, time_steps, im_height=260, im_width=346, im_channel=1, time_window=None):
        #One Channel
        event_images_list = []

        event_iterator = 0
        event_iterator_old = 0
        event_threshold_classifier = 10000
        event_bus = 0
        measure = 0
        for ts in time_steps:
            measure = measure + 1
            event_image = np.zeros((im_height, im_width, im_channel), dtype=np.float32)
            for i in range(event_iterator, len(event_list)):
                event_iterator = i + 1
                event = np.copy(event_list[i]) #[x, y, ts, polarity]
                if event_bus == event_threshold_classifier:
                    event_image[201, 154, 0] = 0
                    event_images_list.append(np.copy(event_image))
                    event_bus = 0
                    break
                else:
                    if event[3] > 0:
                        event_image[int(event[1]), int(event[0]), 0] = event_image[int(event[1]), int(event[0]), 0] + 1.0
                    else:
                        event_image[int(event[1]), int(event[0]), 0] = event_image[int(event[1]), int(event[0]), 0] - 1.0
                    event_bus = event_bus + 1
        print(measure)


        print("event_iterator: ", event_iterator)
        print("event_list length: ", len(event_list)) 
        return event_images_list, event_list[event_iterator:len(event_list)]

    #extractEventsByTime static method
    @staticmethod
    def extractEventsByTime(event_list, time_steps, time_window=1e9):
        #Group Events into based on tghe provided list of timestamps
        grouped_events_list = []

        event_iterator = 0
        for ts in time_steps:
            print("Percentage finished: " + str(event_iterator * 100 / len(event_list)))
            event_group = []
            
            for i in range(event_iterator, len(event_list)):
                event_iterator = i + 1
                event = np.copy(event_list[i])
                if (time_window==None) or (event[2] > ts - time_window/2):
                    event_group.append(event)
                    if event[2] > ts + time_window/2:
                        grouped_events_list.append(np.copy(event_group))
                        break            


        return grouped_events_list

    #extractEventsByTime static method
    @staticmethod
    def eventsToFrameSequence(events, im_height=260, im_width=346, im_channel=1, time_window=0.2e9, N_frames=-1):
        #Group Events into frames, positive frames count positively while negative subtract
        #Channel 1: count of positive events at each pixel position
        #Channel 2: count of negative events at each pixel position
        #Channel 3: normalized timestamp of events generated at pixel position
        event_images_list = []        
        ts_min = events[0][2]

        time_iterator = 1        
        event_image = np.zeros((im_height, im_width, im_channel), dtype=np.float32)
        for i in range(len(events)):
            event = events[i]
            if event[3] > 0:
                event_image[int(event[1]), int(event[0]), 0] = event_image[int(event[1]), int(event[0]), 0] + 1.0
            else:
                event_image[int(event[1]), int(event[0]), 0] = event_image[int(event[1]), int(event[0]), 0] - 1.0   
            

            if (event[2] > (time_iterator * time_window + ts_min)) or (i==(len(events)-1)):
                event_images_list.append(event_image)
                event_image = np.zeros((im_height, im_width, im_channel), dtype=np.float32)   
                time_iterator = time_iterator + 1

        if N_frames > 0:
            while len(event_images_list) < N_frames:
                event_image = np.zeros((im_height, im_width, im_channel), dtype=np.float32)  
                event_images_list.append(event_image)
            
            while len(event_images_list) > N_frames:
                del event_images_list[0]

        


        return event_images_list


    #cropFrames static method
    @staticmethod
    def cropFrames(image_list, circle_center=(173, 130), circle_rad=100, im_height=260, im_width=346, im_channels=1, expand_dims=False):
        
        cropped_image_list = []
        i=0
        for image in image_list:
            mask = np.zeros((im_height, im_width, im_channels), dtype=np.float32)            
            cv2.circle(mask, circle_center, circle_rad, [1]*im_channels, -1, 8, 0)
            cropped_image = np.multiply(mask, image)

            if expand_dims:
                cropped_image_list.append(np.expand_dims(np.copy(cropped_image), axis=2)) 
            else:
                cropped_image_list.append(np.copy(cropped_image)) 

        return cropped_image_list

    #rotateFrames static method
    @staticmethod
    def rotateFrames(image_list, circle_center=(173, 130), rotate_angle=90, im_height=260, im_width=346, expand_dims=False):
        
        rotated_image_list = []

        for image in image_list:
            rot_mat = cv2.getRotationMatrix2D(circle_center, rotate_angle, 1.0)
            #rotated_image = cv2.warpAffine(image[:,:,0], rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
            rotated_image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

            if expand_dims:
                rotated_image_list.append(np.expand_dims(np.copy(rotated_image), axis=2)) 
            else:
                rotated_image_list.append(np.copy(rotated_image))

        return rotated_image_list


    #updateContactStatus static method
    @staticmethod
    def updateContactStatus(list_of_current_contact_status, list_of_rotations, rotation_angle):
        
        list_of_rotated_contact_status = []

        rot_mat = np.array([ [math.cos(rotation_angle*math.pi/180), math.sin(rotation_angle*math.pi/180)], [-math.sin(rotation_angle*math.pi/180), math.cos(rotation_angle*math.pi/180)] ])

        for contact_status in list_of_current_contact_status:
            if contact_status == 0:
                list_of_rotated_contact_status.append(0)
            else:
                current_rotation = list_of_rotations[int(contact_status)-1]
                rotated_contact_status = np.matmul(rot_mat, current_rotation[0:2])
                best_rot_diff = 100
                best_rot_idx = 1
                i = 1
                for rot in list_of_rotations:
                    diff_vals = np.sqrt( np.power(rot[0] - rotated_contact_status[0], 2) +  np.power(rot[1] - rotated_contact_status[1], 2))
                    if best_rot_diff > diff_vals:
                        best_rot_diff = diff_vals
                        best_rot_idx = i
                    i = i + 1

                list_of_rotated_contact_status.append(best_rot_idx)


        return list_of_rotated_contact_status

    
    def EventBinning(event_list, time_steps, im_height=260, im_width=346, n_bin=9, bin_step_size=0.05):
        #Bin event into n_bin, each bin_step_size apart as described in "Unsupervised Event-based Learning of Optical Flow, Depth, and Egomotion"
        #The time of the first bin should correspond to the elements in time_steps

        event_images_list = []

        event_leading_iterator = 0    
        event_trailing_iterator = 0

        for ts in time_steps:
            print("Percentage finished: %d", event_leading_iterator / len(event_list))
            event_image = np.zeros((im_height, im_width, n_bin), dtype=np.float64)
            
            for i in range(event_leading_iterator, len(event_list)):
                event = np.copy(event_list[i])
                if event[2] > ts:
                    event_leading_iterator = i
                    break
            for i in range(event_trailing_iterator, event_leading_iterator):
                event = np.copy(event_list[i])
                if event[2] > (ts - n_bin*bin_step_size):
                    event_trailing_iterator = i
                    break
            for i in range(event_trailing_iterator, event_leading_iterator):
                event = np.copy(event_list[i])
                channel = np.floor((ts - event[2]) / bin_step_size )
                event_image[int(event[1]), int(event[0]), int(channel)] = event_image[int(event[1]), int(event[0]), int(channel)] + event[3]

            event_images_list.append(event_image)


        return event_images_list 

    def SelectFrames(images_list, images_ts, desired_timesteps, target_step=0.06):
        #From a list of timestamped images, select images closest to the desired timestamps

        desired_frames = []
        images_iterator = 0

        for ts in desired_timesteps:
            for i in range(images_iterator, len(images_list)):
                if np.abs(images_ts[i] - ts) < target_step:
                    desired_frames.append(images_list[i])
                    images_iterator = i + 1
                    break
                elif (images_ts[i] - ts) > 1:
                    break

        return(desired_frames)

    def EdgeDetection(image_list, canny_threshold_1=100, canny_threshold_2=100, convert=False):
        #Apply Canny Edge Detection to a list of images
        edge_images = []

        for image in image_list:
            temp_image = np.copy(image)
            temp_image[np.isnan(temp_image)] = 0
            if convert:
                temp_image = np.uint8(255 * (temp_image - np.min(temp_image)) / (np.max(temp_image) - np.min(temp_image)))
            edge_images.append(cv2.Canny(temp_image, canny_threshold_1, canny_threshold_2))

        return edge_images
        
    def ErodeImages(image_list, kernel_size=5, binary_format=False):
        #Erode a list of images
        eroded_images = []
        erosion_kernel = np.ones((kernel_size, kernel_size), np.uint8)

        for image in image_list:
            temp_image = np.copy(image)
            temp_image[np.isnan(temp_image)] = 0.0
            temp_image = 255 * (temp_image - np.min(temp_image)) / (np.max(temp_image) - np.min(temp_image))
            eroded_image = cv2.erode(temp_image, erosion_kernel)
            if binary_format:
                eroded_image[eroded_image>0] = 1
            eroded_images.append(eroded_image)

        return eroded_images

    def DiluteImages(image_list, kernel_size=5, binary_format=False):
        #Dilate a list of images
        diluted_images = []
        dilate_kernel = np.ones((kernel_size, kernel_size), np.uint8)

        for image in image_list:
            temp_image = np.copy(image)
            temp_image[np.isnan(temp_image)] = 0.0
            if binary_format:
                temp_image = 255 * (temp_image - np.min(temp_image)) / (np.max(temp_image) - np.min(temp_image))
            diluted_image = cv2.dilate(temp_image, dilate_kernel)
            if binary_format:
                diluted_image[diluted_image>0] = 1
            diluted_images.append(diluted_image)

        return diluted_images

    def BinaryAddImages(image_list_1, image_list_2):
        #Binary add two list of images
        summed_images = []
        size = np.min([len(image_list_1), len(image_list_2)])
        
        for i in range(size):
            summed_images.append(np.logical_or(image_list_1[i], image_list_2[i]))
        
        return summed_images

    def MultiplyImages(image_list_1, image_list_2):
        #Multiply two list of images (for masking)

        multiplied_images = []
        size = np.min([len(image_list_1), len(image_list_2)])
        for i in range(size):
            multiplied_images.append(np.multiply(image_list_1[i], image_list_2[i]))
        
        return multiplied_images


    def FlipImages(image_list, axis=0):
        #flip image vertically (axis=0) or horizontally (axis=1)

        flipped_image_list = []

        for image in image_list:
            flipped_image_list.append(np.flip(image, axis))

        return flipped_image_list