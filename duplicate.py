import cv2

def orb_similarity(img1, img2):
    orb = cv2.ORB_create()

    # Find keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return 0

    # Match features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    return len(matches)

def are_images_similar_orb(img1, img2, match_threshold=30):
    matches = orb_similarity(img1, img2)
    return matches > match_threshold
