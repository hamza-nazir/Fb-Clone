import { useEffect } from "react";
import { createContext } from "react";
import { useState } from "react";
import api from '../Hook/Axios'
export const Context=createContext()

export const ReduxFun=({children})=>{
const [userDetails,setUserDetails]=useState({});
const [userPosts,setUserPosts]=useState([])
   const [option,setOption]=useState('feed');
   const [commentsData,setCommentsData]=useState([])
const [userRelation,setUserRelation]=useState({})
        const [posts,setPosts]=useState([])
    const [selectedPost, setSelectedPost] = useState({});

      const [comment,setComment]=useState(false);

      const [open,setOpen]=useState(false);
            const [edit,setEdit]=useState(false);
            const [percentage,setPercentage]=useState(0)
      const [postPopUp,setPostPopUp]=useState(false);
          const [logout,setLogout]=useState(false);
           const [request,setRequest]=useState(false);
              const [profileOption,setProfileOption]=useState("posts");
              
    return(
    
    <Context.Provider value={{percentage,setPercentage,commentsData,setCommentsData,selectedPost, setSelectedPost,posts,setPosts,comment,setComment,userRelation,setUserRelation,userPosts,setUserPosts,option,setOption,open,setOpen,postPopUp,setPostPopUp,logout,setLogout,request,setRequest,edit,setEdit,userDetails,setUserDetails,profileOption,setProfileOption}} >
        {children}
    </Context.Provider>
    )
  
}