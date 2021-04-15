import vsrl_utils as vu
import numpy as np
import pandas as pd



class MyVCOCO:
    def __init__(self, type:str):
        # Load COCO annotations for V-COCO images
        self.coco = vu.load_coco()
        self.vcoco = vu.load_vcoco(type)
        for x in self.vcoco:
            x = vu.attach_gt_boxes(x, self.coco)

        self.role_list = None

    def getTarget(self, vcoco) -> np.array:
        target = vcoco['label'].ravel()
        target = np.where(target == 1)[0]
        return target

    def getAction(self, vcoco) -> np.array:
        return vcoco['action_name']

    def getPersonList(self, vcoco, target):
        return vcoco['role_object_id'][target][:,0]


    def getRoleObjExPerson(self, vcoco, target):
        return vcoco['role_object_id'][target][:,1:]


    def getRoleId2ObjName(self, role_list):
        def roleID2objID(x):
            if x != 0:
                name = self.coco.cats[self.coco.loadAnns(x)[0]['category_id']]['name']
                return name
            return 'no-obj'

        def roleID2Bbox(x):
            if x != 0:
                name = self.coco.loadAnns(x)[0]['bbox']
                return name
            return 'no-bbox'

        role_list['obbox'] = role_list['obj_id'].apply(roleID2Bbox)
        role_list['obj'] = role_list['obj_id'].apply(roleID2objID)
        return role_list


    def getRoleNameExPerson(self, vcoco, target):
        return vcoco['role_name'][1:]


    def getImageID(self, vcoco, target):
        img_id = vcoco['image_id'][target].ravel().tolist()
        img_id = self.coco.loadImgs(img_id)
        # print(img_id)
        # img_id = [id['file_name'] for id in img_id]
        img_id = [id['flickr_url'] for id in img_id]
        return img_id


    def getPerson2Pbbox(self, role_list):

        def personID2Bbox(x):
            # print(x)
            if x != 0:
                name = self.coco.loadAnns(x)[0]['bbox']
                return name
            return 'no-bbox'

        role_list['pbbox'] = role_list['person_id'].apply(personID2Bbox)


        return role_list

    def getRoleList(self, ret, vcoco, target):
        person_list = self.getPersonList(vcoco, target)
        # print(person_list)
        role_list = self.getRoleObjExPerson(vcoco, target) # 사람 객체 제외
        role_names = self.getRoleNameExPerson(vcoco, target) # 사람 객체 제외
        action = self.getAction(vcoco)
        img_ids = self.getImageID(vcoco, target)


        for ind in range(len(role_names)):
            _ret = pd.DataFrame()

            act = action
            act += '-'
            act += role_names[ind]
            obj = role_list[:,ind]

            # print(act)

            # print(val)

            _ret['obj_id'] = obj
            _ret['action'] = act
            _ret['img_id'] = img_ids
            _ret['person_id'] = person_list
            ret = ret.append(_ret, ignore_index=True)

        # print(ret)
        return ret

    def maskDF(self):
        self.role_list = pd.DataFrame()
        for vcoco_per_verb in self.vcoco:
            # print(vcoco_per_verb)
            # print(coco_per_verb.keys())
            target = self.getTarget(vcoco_per_verb)


            self.role_list = self.getRoleList(self.role_list, vcoco_per_verb, target)

        self.role_list = self.getRoleId2ObjName(self.role_list)
        self.role_list = self.getPerson2Pbbox(self.role_list)
        # print(role_list)

        return self.role_list



if __name__ == '__main__':
    train_vcoco = MyVCOCO('vcoco_train')
    train_df = train_vcoco.maskDF()
    print(train_df)
    # print()