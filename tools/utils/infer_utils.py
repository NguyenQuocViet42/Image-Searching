def infer_one_image(image, detector, tracker):
    outputs = {'detect': None, 'tracking': None}
    if detector is not None:
        out_detect = detector([image])
        outputs['detect'] = out_detect
        if tracker is not None:
            out_track = tracker.update(out_detect[0][0], out_detect[1][0], out_detect[2][0])
            outputs['tracking'] = out_track

    return outputs