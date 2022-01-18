# 二叉树篇

### 23.二叉树

#### 23.1 二叉树的前序遍历

```java
public class TreeNode{
    int val;
    TreeNode left;
    TreeNode right;
    TreeNode(int val){
        this.val = val;
    }
}
//非递归
public void preOrder(TreeNode root){
    if(root == null) return;
    Stack<TreeNode> stack = new Stack<>();
    stack.push(root);
    while(!stack.isEmpty()){
        TreeNode node = stack.pop();
        System.out.print(node.val);
        //因为栈是先进后出，所以先压右孩子，再压左孩子
        if(node.right != null)
            stack.push(node.right);
        if(node.left != null)
            stack.push(node.left);
    } 
}
//递归
public void preOrder(TreeNode root){
    if(root == null) return;
    System.out.print(root.val);
    preOrder(root.left);
    preOrder(root.right);
}

```

#### 23.2 二叉树的后序遍历

```java
public void postOrder(TreeNode root){
    if(root == null) return;
    Stack<TreeNode> s1 = new Stack<>();
    Stack<TreeNode> s2 = new Stack<>();
    s1.push(root);
    while(!s1.isEmpty()){
        TreeNode node = s1.pop();
        s2.push(node);
        if(node.left != null)
           s1.push(node.left);
       if(node.right != null)
           s1.push(node.right);
    }
    while(!s2.isEmpty())
        System.out.print(s2.pop().val + " ");
}
```

#### 23.3 二叉树的中序遍历

```java
//中序遍历非递归
class Solution {
    public List<Integer> inorderTraversal(TreeNode root) {
        Stack<TreeNode> stack = new Stack<>();
        TreeNode p =root;
        ArrayList<Integer> list = new ArrayList<>();
        while(p != null || !stack.isEmpty()){
            //一直走到最左边的左孩子
            while(p != null){
                stack.push(p);
                p = p.left;
            }
            p = stack.pop();
            list.add(p.val);
            p = p.right;
        }
        return list;
    }
}
```

#### 23.4 从上往下打印二叉树(层序遍历)

```java
public class TreeNode {
    int val;
    TreeNode left = null;
    TreeNode right = null;
    public TreeNode(int val){
        this.val = val;
    }
}

public class Solution {
    public ArrayList<Integer> PrintFromTopToBottom(TreeNode root) {
        ArrayList<Integer> result = new ArrayList<>();
        if(root == null)
            return result;
        LinkedList<Integer> queue = new LinkedList<>();
        queue.add(root);
        while(!queue.isEmpty()){
            TreeNode node = queue.poll();
            result.add(node.val);
            if(node.left != null){
                queue.add(node.left);
            }
            if(node.right != null){
                queue.add(node.right);
            }
        }
        return result;
    }
}
```

#### 23.5 蛇形遍历二叉树

与层序遍历的区别是当层数为奇数就从左往右遍历，层数是偶数，就从右往左。

```java
public class Solution {
    public ArrayList<Integer> PrintFromTopToBottom(TreeNode root) {
        ArrayList<Integer> result = new ArrayList<>();
        if(root == null)
            return result;
        LinkedList<Integer> queue = new LinkedList<>();
        queue.add(root);
        int level = 1;//认为根节点在第一层
        while(!queue.isEmpty()){
            TreeNode node = queue.poll();
            result.add(node.val);
            level++;
            if(level % 2 == 1){
                if(node.left != null){
                    queue.add(node.left);
                }
                if(node.right != null){
                    queue.add(node.right);
                }
            }else {
                if(node.right != null){
                    queue.add(node.right);
                }
                if(node.left != null){
                    queue.add(node.left);
                }
            }
           
        }
        return result;
    }
}
```

#### 23.6 二叉搜索树的第k大的节点

```java
//迭代：
class Solution {
    public int kthLargest(TreeNode root, int k) {
        Stack<TreeNode> stack = new Stack<>();
        TreeNode p = root;
        int count = 0;
        while(p != null || !stack.isEmpty()){
            while(p != null){
                stack.push(p);
                p = p.right;
            }
            p = stack.pop();
            count++;
            if(count == k){
                return p.val;
            }
            p = p.left;
        }
        return 0;
    }
}
//递归
class Solution {
    int res,index;
    public int kthLargest(TreeNode root, int k) {
        index = k;
        dfs(root);
        return res;
    }
    public void dfs(TreeNode root){
        if(root == null) return;
        dfs(root.right);
        if(index == 0) return;
        index--;
        if(index == 0) res = root.val;
        dfs(root.left);
    }
}

```

#### 23.7 求完全二叉树的节点个数

完全[二叉树](https://www.nowcoder.com/jump/super-jump/word?word=二叉树)的定义如下：在完全[二叉树](https://www.nowcoder.com/jump/super-jump/word?word=二叉树)中，除了最底层节点可能没填满外，其余每层节点数都达到最大值，并且最下面一层的节点都集中在该层最左边的若干位置。若最底层为第 h 层，则该层包含 1~2^h个节点。

```java
class Solution {
    public int countNodes(TreeNode root) {
        if(root == null)
            return 0;
        int left = countlevel(root.left);
        int right = countlevel(root.right);
        if(left == right){
            //移位运算，1向左移动left次，相当于1*2^n
            return countNodes(root.right) + (1<<left);
        }else{
            return countNodes(root.left) + (1<<right);
        }     
    }
    private int countlevel(TreeNode root){
        int level = 0;
        while(root != null){
            level++;
            root = root.left;
        }
        return level;  
    }
}
```

#### 23.8 求二叉树根节点到叶节点的路径和的最小值

递归：只有左子树，就只计算左子树，只有右子树，就只计算右子树，两个都有，就计算左右子树取最小值。

```java
public int minPath(TreeNode root){
    if(root == null) return 0;
    if(root.left != null && root.right == null){
        return 1+minPath(root.left);
    }
    if(root.left == null && root.right != null){
        return 1 + minPath(root.right);
    }
    return 1 + Math.min(minPath(root.left),minPath(root.right));
}
```

#### 23.9 二叉树的最大路径和

路径 被定义为一条从树中任意节点出发，沿父节点-子节点连接，达到任意节点的序列。同一个节点在一条路径序列中 至多出现一次 。该路径 至少包含一个 节点，且不一定经过根节点。

路径和 是路径中各节点值的总和。

给你一个二叉树的根节点 root ，返回其 最大路径和 。

```java
       int max = Integer.MIN_VALUE;
    public int maxPathSum(TreeNode root) {
        if (root == null) {
            return 0;
        }
        dfs(root);
        return max;
    }

    /**
     * 返回经过root的单边分支最大和， 即Math.max(root, root+left, root+right)
     * @param root
     * <a href="/profile/547241" data-card-uid="547241" class="js-nc-card" target="_blank" from-niu="default">@return */
    public int dfs(TreeNode root) {
        if (root == null) {
            return 0;
        }
        //计算左边分支最大值，左边分支如果为负数还不如不选择
        int leftMax = Math.max(0, dfs(root.left));
        //计算右边分支最大值，右边分支如果为负数还不如不选择
        int rightMax = Math.max(0, dfs(root.right));
        //left->root->right 作为路径与历史最大值做比较
        max = Math.max(max, root.val + leftMax + rightMax);
        // 返回经过root的单边最大分支给上游
        return root.val + Math.max(leftMax, rightMax);
    }    </a>
```

#### 23.10 Leetcode687 最长同值路径

给定一个二叉树，找到最长的路径，这个路径中的每个节点具有相同值。 这条路径可以经过也可以不经过根节点。

注意：两个节点之间的路径长度由它们之间的边数表示。

示例 1:

输入:

              5
             / \
            4   5
           / \   \
          1   1   5



```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    // 最长同值路径
    int max_path = 0;
    public int longestUnivaluePath(TreeNode root) {
        if(root==null){
            return max_path;
        }
        height(root,root.val);
        return max_path;
    }
    public int height(TreeNode root,int val){
        if(root==null){
            return 0;
        }
        int left = height(root.left,root.val);
        int right = height(root.right,root.val);

        max_path = Math.max(max_path,left+right);
        if(root.val==val){
            return Math.max(left,right)+1;
        }else{
            return 0;
        }

    }
}
```



#### 23.10 多叉树的最近公共祖先

 **二叉树的最近公共祖先**

```java
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if(root == null) return null;
        if(root == p || root == q) return root;
        TreeNode left = lowestCommonAncestor(root.left,p,q);
        TreeNode right = lowestCommonAncestor(root.right,p,q);
        if(left == null) return right;
        if(right == null) return left;
        if(right != null && left != null) return root;
        return null;
    }
}
```

#### 23.11 路径总和

给你二叉树的根节点 root 和一个表示目标和的整数 targetSum ，判断该树中是否存在 根节点到叶子节点 的路径，这条路径上所有节点值相加等于目标和 targetSum 。

叶子节点 是指没有子节点的节点。

**示例 1：**

![img](https://assets.leetcode.com/uploads/2021/01/18/pathsum1.jpg)

```
输入：root = [5,4,8,11,null,13,4,7,2,null,null,null,1], targetSum = 22
输出：true
```

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    //判断从根节点到叶子结点是否存在这样的路径 类似两子树之类的
    public boolean hasPathSum(TreeNode root, int targetSum) {
        if(root==null){
            return false;
        }
        // 根左右
        if(root.left==null&&root.right==null){
            return root.val==targetSum;
        }
        // 继续判断
        return hasPathSum(root.left,targetSum-root.val)||hasPathSum(root.right,targetSum-root.val);
    }
}
```

#### 23.12 路径总和III

给定一个二叉树，它的每个结点都存放着一个整数值。

找出路径和等于给定数值的路径总数。

路径不需要从根节点开始，也不需要在叶子节点结束，但是路径方向必须是向下的（只能从父节点到子节点）。

二叉树不超过1000个节点，且节点数值范围是 [-1000000,1000000] 的整数。

示例：

root = [10,5,-3,3,2,null,11,3,-2,null,1], sum = 8

      10
     /  \
    5   -3

   / \    \
  3   2   11
 / \   \
3  -2   1

返回 3。和等于 8 的路径有:

1.  5 -> 3
2.  5 -> 2 -> 1
3.  -3 -> 11

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    // 全部遍历dfs
    int sum = 0;
    public int pathSum(TreeNode root, int targetSum) {
        if(root==null){
            return sum;
        }
        dfs(root,targetSum);
        pathSum(root.left,targetSum);
        pathSum(root.right,targetSum);
        return sum;
    }

    public void dfs(TreeNode root,int targetSum){
        if(root==null){
            return;
        }
        targetSum-=root.val;
        if(targetSum==0){
            sum++;
        }

        dfs(root.left,targetSum);
        dfs(root.right,targetSum);
    }
}
```

#### 23.13 路径总和II

给你二叉树的根节点 root 和一个整数目标和 targetSum ，找出所有 从根节点到叶子节点 路径总和等于给定目标和的路径。

叶子节点 是指没有子节点的节点。

 

示例 1：

![img](https://assets.leetcode.com/uploads/2021/01/18/pathsumii1.jpg)


输入：root = [5,4,8,11,null,13,4,7,2,null,null,5,1], targetSum = 22
输出：[[5,4,11,2],[5,8,4,5]]

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    // dfs解题思路
    List<List<Integer>> res = new ArrayList<>();
    public List<List<Integer>> pathSum(TreeNode root, int targetSum) {
        if(root==null){
            return res;
        }
        List<Integer> path = new ArrayList<>();
        dfs(root,targetSum,path);
        return res;
    }

    public void dfs(TreeNode root,int targetSum,List<Integer> path){
        //递归截止条件
        if(root==null){
            return;
        }
        // 根左右
        path.add(root.val);
        targetSum-=root.val;
        //处理
        if(root.left==null&&root.right==null&&targetSum==0){
            res.add(new ArrayList<>(path));
        }

        dfs(root.left,targetSum,path);
        dfs(root.right,targetSum,path);
        path.remove(path.size()-1);
    }
}
```

#### 23.14 恢复二叉树

给你二叉搜索树的根节点 root ，该树中的两个节点被错误地交换。请在不改变其结构的情况下，恢复这棵树。

进阶：使用 O(n) 空间复杂度的解法很容易实现。你能想出一个只使用常数空间的解决方案吗？

 

示例 1：

![img](https://assets.leetcode.com/uploads/2020/10/28/recover1.jpg)


输入：root = [1,3,null,null,2]
输出：[3,1,null,null,2]
解释：3 不能是 1 左孩子，因为 3 > 1 。交换 1 和 3 使二叉搜索树有效。

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public void recoverTree(TreeNode root) {
        //恢复二叉搜索树非递归的形式
        Stack<TreeNode> stack = new Stack<>();
        TreeNode pre = null;
        TreeNode first = null;
        TreeNode second = null;
        // 非递归的形式
        while(!stack.isEmpty()||root!=null){
            while(root!=null){
                stack.push(root);
                root = root.left;
            }
            root = stack.pop();
            // 处理并且不符合条件
            if(pre!=null&&pre.val>root.val){
                second = root;
                if(first==null){
                    first = pre;
                }else{
                    break;
                }
            }
            // 记录
            pre = root;
            root = root.right;
        }
        // 交换
        swap(first,second);
    }

    public void swap(TreeNode first,TreeNode second){
        int temp = first.val;
        first.val = second.val;
        second.val = temp;
    }
}
```

### 

### 【1.Leetcode104 二叉树的最大深度】

给定一个二叉树，找出其最大深度。

二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。

说明: 叶子节点是指没有子节点的节点。

示例：
给定二叉树 [3,9,20,null,null,15,7]，

    3
   / \
  9  20
    /  \
   15   7
返回它的最大深度 3 。

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public int maxDepth(TreeNode root) {
        if(root==null){
            return 0;
        }
        int leftHeight = maxDepth(root.left);
        int rightHeight = maxDepth(root.right);
        int height = Math.max(leftHeight,rightHeight)+1;
        return height;
    }
}
```

### 【2.Leetcode102 二叉树的层序遍历】

给你一个二叉树，请你返回其按 层序遍历 得到的节点值。 （即逐层地，从左到右访问所有节点）。

 

示例：
二叉树：[3,9,20,null,null,15,7],

3

   / \
  9  20
    /  \
   15   7
返回其层序遍历结果：

[
  [3],
  [9,20],
  [15,7]
]

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        // BFS解题算法
        List<List<Integer>> res = new ArrayList<>();
        if(root==null){
            return res;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        List<Integer> list = new ArrayList<>();
        queue.offer(root);
        int levelSize = 1;
        while(!queue.isEmpty()){
            root = queue.poll();
            levelSize--;
            list.add(root.val);

            if(root.left!=null){
                queue.offer(root.left);
            }

            if(root.right!=null){
                queue.offer(root.right);
            }

            if(levelSize==0){
                levelSize = queue.size();
                res.add(new ArrayList<>(list));
                list = new ArrayList<>();
            }
        }
        return res;
    }
}
```

### 【3.Leetcode101 对称二叉树】

给定一个二叉树，检查它是否是镜像对称的。

 

例如，二叉树 [1,2,2,3,4,4,3] 是对称的。

1

   / \
  2   2
 / \ / \
3  4 4  3


但是下面这个 [1,2,2,null,3,null,3] 则不是镜像对称的:

1

   / \
  2   2
   \   \
   3    3

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public boolean isSymmetric(TreeNode root) {
        if(root==null){
            return true;
        }
        return judge(root.left,root.right);
    }
    public boolean judge(TreeNode t1,TreeNode t2){
        // 都为空 走到底了
        if(t1==null&&t2==null){
            return true;
        }
        // 有一个有数 有一个没有
        if(t1==null||t2==null){
            return false;
        }
        // 判断
        if(t1.val!=t2.val){
            return false;
        }

        return judge(t1.left,t2.right)&&judge(t1.right,t2.left);
    }
}
```

### 【4.Leetcode94 二叉树的中序遍历】


给定一个二叉树的根节点 `root` ，返回它的 **中序** 遍历。

 

**示例 1：**

![img](https://assets.leetcode.com/uploads/2020/09/15/inorder_1.jpg)

```
输入：root = [1,null,2,3]
输出：[1,3,2]
```

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    // 结果
    List<Integer> res = new ArrayList<>();
    public List<Integer> inorderTraversal(TreeNode root) {
        if(root==null){
            return res;
        }
        inorderTraversal(root.left);
        res.add(root.val);
        inorderTraversal(root.right);

        return res;
    }
}
```

> 非递归的形式

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        // 栈
        Stack<TreeNode> stack = new Stack<>();
        // 中序遍历 左根右
        while(!stack.isEmpty()||root!=null){
            // 不为空
            while(root!=null){
                stack.push(root);
                root = root.left;
            }
            root = stack.pop();
            res.add(root.val);
            root = root.right;
        }
        return res;
    }
}
```

### 【5.Leetcode98 验证二叉搜索树】

给定一个二叉树，判断其是否是一个有效的二叉搜索树。

假设一个二叉搜索树具有如下特征：

节点的左子树只包含小于当前节点的数。
节点的右子树只包含大于当前节点的数。
所有左子树和右子树自身必须也是二叉搜索树。
示例 1:

输入:
    2
   / \
  1   3
输出: true
示例 2:

输入:
    5
   / \
  1   4
     / \
    3   6
输出: false
解释: 输入为: [5,1,4,null,null,3,6]。
     根节点的值为 5 ，但是其右子节点值为 4 。

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    long prev = Long.MIN_VALUE;
    public boolean isValidBST(TreeNode root) {
        // 验证二叉搜索树是否合法有效
        if(root==null){
            return true;
        }
        boolean left = isValidBST(root.left);

        // 判断是否合法
        if(prev!=Long.MIN_VALUE&&prev>=root.val){
            return false;
        }
        prev = root.val;

        boolean right = isValidBST(root.right);

        return left&&right;
    }
}
```

###  【6.Leetcode100 相同的数】

给你两棵二叉树的根节点 p 和 q ，编写一个函数来检验这两棵树是否相同。

如果两个树在结构上相同，并且节点具有相同的值，则认为它们是相同的。

 

示例 1：

![img](https://assets.leetcode.com/uploads/2020/12/20/ex1.jpg)


输入：p = [1,2,3], q = [1,2,3]
输出：true

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public boolean isSameTree(TreeNode p, TreeNode q) {
        if(p==null&&q==null){
            return true;
        }

        if(p==null || q==null){
            return false;
        }

        if(p.val!=q.val){
            return false;
        }

        return isSameTree(p.left,q.left)&&isSameTree(p.right,q.right);

    }
}
```

### 【7.Leetcode226 翻转二叉树】

翻转一棵二叉树。

示例：

输入：

 4

   /   \
  2     7
 / \   / \
1   3 6   9
输出：

 4

   /   \
  7     2
 / \   / \
9   6 3   1

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public TreeNode invertTree(TreeNode root) {
        if(root==null){
            return root;
        }   
        TreeNode temp = root.left;
        root.left = root.right;
        root.right = temp;
        
        invertTree(root.left);
        invertTree(root.right);

        return root;

    }
}
```

### 【8.Leetcode111 二叉树的最小深度】

给定一个二叉树，找出其最小深度。

最小深度是从根节点到最近叶子节点的最短路径上的节点数量。

说明：叶子节点是指没有子节点的节点。

 

示例 1：

![img](https://assets.leetcode.com/uploads/2020/10/12/ex_depth.jpg)


输入：root = [3,9,20,null,null,15,7]
输出：2



```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public int minDepth(TreeNode root) {
        // 从根节点到叶子节点的最短路径上的结点数量
        if(root==null){
            return 0;
        }
        int leftHeight = minDepth(root.left);
        int rightHeight = minDepth(root.right);

        return (leftHeight==0||rightHeight==0)?leftHeight+rightHeight+1:Math.min(leftHeight,rightHeight)+1;
    }
}
```

### 【9.Leetcode144 二叉树的前序遍历】

给你二叉树的根节点 `root` ，返回它节点值的 **前序** 遍历。

 

**示例 1：**

![img](https://assets.leetcode.com/uploads/2020/09/15/inorder_1.jpg)

```
输入：root = [1,null,2,3]
输出：[1,2,3]
```

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    List<Integer> res = new ArrayList<>();
    public List<Integer> preorderTraversal(TreeNode root) {
        if(root==null){
            return res;
        }
        res.add(root.val);
        preorderTraversal(root.left);
        preorderTraversal(root.right);
        return res;
    }
}
```

> 非递归的形式

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    List<Integer> res = new ArrayList<>();
    public List<Integer> preorderTraversal(TreeNode root) {
        Stack<TreeNode> stack = new Stack<>();
        if(root==null){
            return res;
        }
        // 开始
        stack.push(root);
        while(!stack.isEmpty()){
            root = stack.pop();
            res.add(root.val);

            if(root.right!=null){
                stack.push(root.right);
            }

            if(root.left!=null){
                stack.push(root.left);
            }
        }
        return res;
    }
}
```

### 【10.Leetcode103 二叉树的锯齿形层序遍历】


给定一个二叉树，返回其节点值的锯齿形层序遍历。（即先从左往右，再从右往左进行下一层遍历，以此类推，层与层之间交替进行）。

例如：
给定二叉树 `[3,9,20,null,null,15,7]`,

```
    3
   / \
  9  20
    /  \
   15   7
```

返回锯齿形层序遍历如下：

```
[
  [3],
  [20,9],
  [15,7]
]
```

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        // 结果
        List<List<Integer>> res = new ArrayList<>();
        if(root==null){
            return res;
        }
        List<Integer> list = new ArrayList<>();
        // 层序遍历
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        int levelSize = 1;
        boolean flag = false;
        // 开始
        while(!queue.isEmpty()){
            root = queue.poll();
            levelSize--;
            list.add(root.val);

            if(root.left!=null){
                queue.offer(root.left);
            }

            if(root.right!=null){
                queue.offer(root.right);
            }

            if(levelSize==0){
                levelSize = queue.size();
                if(flag){
                    Collections.reverse(list);
                }
                flag = !flag;
                res.add(list);
                list = new ArrayList<>();
            }
        }
        return res;
    }
}
```

### 【11.Leetcode112 路径总和】

给你二叉树的根节点 root 和一个表示目标和的整数 targetSum ，判断该树中是否存在 根节点到叶子节点 的路径，这条路径上所有节点值相加等于目标和 targetSum 。

叶子节点 是指没有子节点的节点。

 

示例 1：

![img](https://assets.leetcode.com/uploads/2021/01/18/pathsum1.jpg)


输入：root = [5,4,8,11,null,13,4,7,2,null,null,null,1], targetSum = 22
输出：true

> 从根节点到叶子结点的路径和

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public boolean hasPathSum(TreeNode root, int targetSum) {
        if(root==null){
            return false;
        }

        // 判断
        if(root.left==null&&root.right==null){
            return targetSum==root.val;
        }

        // 继续
        return hasPathSum(root.left,targetSum-root.val) || hasPathSum(root.right,targetSum-root.val);
    }
}
```

### 【12.Leetcode113 路径总和II】

给你二叉树的根节点 root 和一个整数目标和 targetSum ，找出所有 从根节点到叶子节点 路径总和等于给定目标和的路径。

叶子节点 是指没有子节点的节点。

 

示例 1：

![img](https://assets.leetcode.com/uploads/2021/01/18/pathsumii1.jpg)


输入：root = [5,4,8,11,null,13,4,7,2,null,null,5,1], targetSum = 22
输出：[[5,4,11,2],[5,8,4,5]]

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    // 结果
    List<List<Integer>> res = new ArrayList<>();
    List<Integer> path = new ArrayList<>();
    public List<List<Integer>> pathSum(TreeNode root, int targetSum) {
        if(root==null){
            return res;
        }
        // dfs
        dfs(root,targetSum);
        return res;
    }
    //dfs的解题思路
    public void dfs(TreeNode root,int targetSum){
        if(root==null){
            return;
        }

        path.add(root.val);

        targetSum -= root.val;
        if(root.left==null&&root.right==null&&targetSum==0){
            res.add(new ArrayList<>(path));
        }

        dfs(root.left,targetSum);
        dfs(root.right,targetSum);
        
        path.remove(path.size()-1);
    }
}
```

### 【13.Leetcode199 二叉树的右视图】

给定一棵二叉树，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。

示例:

输入: [1,2,3,null,5,null,4]
输出: [1, 3, 4]
解释:

   1            <---
 /   \
2     3         <---
 \     \
  5     4       <---

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public List<Integer> rightSideView(TreeNode root) {
        // 结果
        List<Integer> res = new ArrayList<>();
        if(root==null){
            return res;
        }
        // 继续
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        int levelSize = 1;
        while(!queue.isEmpty()){
            root = queue.poll();
            if(levelSize==1){
                res.add(root.val);
            }
            levelSize--;

            if(root.left!=null){
                queue.offer(root.left);
            }

            if(root.right!=null){
                queue.offer(root.right);
            }

            if(levelSize==0){
                levelSize = queue.size();
                
            }
        }
        return res;
    }
}
```

### 【14.Leetcode145 二叉树的后序遍历】

给定一个二叉树，返回它的 后序 遍历。

示例:

输入: [1,null,2,3]  
   1
    \
     2
    /
   3 

输出: [3,2,1]

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    // 结果
    List<Integer> res = new ArrayList<>();
    public List<Integer> postorderTraversal(TreeNode root) {
        if(root==null){
            return res;
        }
        postorderTraversal(root.left);
        postorderTraversal(root.right);
        res.add(root.val);

        return res;
    }
}
```

### 【15.Leetcode110 平衡二叉树】

给定一个二叉树，判断它是否是高度平衡的二叉树。

本题中，一棵高度平衡二叉树定义为：

一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过 1 。

 

示例 1：

![img](https://assets.leetcode.com/uploads/2020/10/06/balance_1.jpg)


输入：root = [3,9,20,null,null,15,7]
输出：true

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public boolean isBalanced(TreeNode root) {
        // 平衡二叉树
        return height(root)>=0;
    }

    // 如果不符合条件就让其为-1
    public int height(TreeNode root){
        if(root==null){
            return 0;
        }
        int leftHeight = height(root.left);
        int rightHeight = height(root.right);
        // 判断
        if(leftHeight==-1 || rightHeight==-1 || Math.abs(leftHeight-rightHeight)>1){
            return -1;
        }else{
            return Math.max(leftHeight,rightHeight)+1;
        }
    }
}
```

### 【16.Leetcode_剑指Offer07 重建二叉树】

输入某二叉树的前序遍历和中序遍历的结果，请重建该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。

 

例如，给出

前序遍历 preorder = [3,9,20,15,7]
中序遍历 inorder = [9,3,15,20,7]
返回如下的二叉树：

3

   / \
  9  20
    /  \
   15   7

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    // 根据前序和中序遍历恢复二叉树
    HashMap<Integer,Integer> dict = new HashMap<>();
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        // 对中序遍历
        for(int i=0;i<inorder.length;i++){
            dict.put(inorder[i],i);
        }
        // 调用
        return getMyTree(preorder,0,preorder.length-1,inorder,0,inorder.length-1);
    }
    
    public TreeNode getMyTree(int[] preorder,int preorder_start,int preorder_end,int[] inorder,int inorder_start,int inorder_end){
        // 递归截止条件
        if(preorder_start>preorder_end){
            return null;
        }
        int root_value = preorder[preorder_start];
        
        int inorder_root_index = dict.get(root_value);
        int len = inorder_root_index-inorder_start;

        TreeNode root = new TreeNode(root_value);
        root.left = getMyTree(preorder, preorder_start+1,    preorder_start+len, inorder,  inorder_start,         inorder_root_index);
        root.right = getMyTree(preorder,preorder_start+len+1,preorder_end,       inorder,   inorder_root_index+1, inorder_end);
        return root;
    }

}
```

### 【17.Leetcode107 二叉树的层序遍历 II】

给定一个二叉树，返回其节点值自底向上的层序遍历。 （即按从叶子节点所在层到根节点所在的层，逐层从左向右遍历）

例如：
给定二叉树 [3,9,20,null,null,15,7],

3

   / \
  9  20
    /  \
   15   7
返回其自底向上的层序遍历为：

[
  [15,7],
  [9,20],
  [3]
]

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public List<List<Integer>> levelOrderBottom(TreeNode root) {
        // 结果
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> list = new ArrayList<>();
        // 层序
        if(root==null){
            return res;
        }
        // 继续
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        int levelSize = 1;
        while(!queue.isEmpty()){
            root = queue.poll();
            list.add(root.val);
            levelSize--;

            if(root.left!=null){
                queue.offer(root.left);
            }

            if(root.right!=null){
                queue.offer(root.right);
            }

            if(levelSize==0){
                levelSize = queue.size();
                res.add(new ArrayList<>(list));
                list = new ArrayList<>();
            }
        }
        Collections.reverse(res);
        return res;
    }
}
```

### 【18.Leetcode226  二叉树的最近公共祖先】

给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。

百度百科中最近公共祖先的定义为：“对于有根树 T 的两个节点 p、q，最近公共祖先表示为一个节点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”

 

示例 1：


输入：root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
输出：3
解释：节点 5 和节点 1 的最近公共祖先是节点 3 。

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        // 最近公共祖先
        if(root==null || root==p || root==q){
            return root;
        }
        // 找其祖先
        TreeNode left = lowestCommonAncestor(root.left,p,q);
        TreeNode right = lowestCommonAncestor(root.right,p,q);
        // 不存在
        if(left==null&&right==null){
            return null;
        }
        if(left==null){
            return right;
        }
        if(right==null){
            return left;
        }
        // 都不为空
        return root;
    }
}
```

### 【19.Leetcode245 二叉搜索树的最近公共祖先】

给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。

百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”

例如，给定如下二叉搜索树:  root = [6,2,8,0,4,7,9,null,null,3,5]

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/14/binarysearchtree_improved.png)

 

示例 1:

输入: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8
输出: 6 
解释: 节点 2 和节点 8 的最近公共祖先是 6。

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */

class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        // 二叉搜索树
        TreeNode cur = root;
        while(true){
            if(cur.val>p.val&&cur.val>q.val){
                cur = cur.left;
            }else if(cur.val<p.val&&cur.val<q.val){
                cur = cur.right;
            }else{
                break;
            }
        }
        return cur;
    }
}
```

### 【20.Leetcode_剑指Offer27 二叉树的镜像】

请完成一个函数，输入一个二叉树，该函数输出它的镜像。

例如输入：

 4

   /   \
  2     7
 / \   / \
1   3 6   9
镜像输出：

 4

   /   \
  7     2
 / \   / \
9   6 3   1

 

示例 1：

输入：root = [4,2,7,1,3,6,9]
输出：[4,7,2,9,6,3,1]

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public TreeNode mirrorTree(TreeNode root) {
        if(root==null){
            return null;
        }
        TreeNode temp = root.left;
        root.left     = root.right;
        root.right    = temp;

        mirrorTree(root.left);
        mirrorTree(root.right);;
        return root;
    }
}
```

### 【21.Leetcode543 二叉树的直径】

给定一棵二叉树，你需要计算它的直径长度。一棵二叉树的直径长度是任意两个结点路径长度中的最大值。这条路径可能穿过也可能不穿过根结点。

 

示例 :
给定二叉树

          1
         / \
        2   3
       / \     
      4   5    
返回 3, 它的长度是路径 [4,2,1,3] 或者 [5,2,1,3]。

 

注意：两结点之间的路径长度是以它们之间边的数目表示。

> 求高度的时候，可以求直径

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    int maxLen = 0;
    public int diameterOfBinaryTree(TreeNode root) {
        height(root);
        return maxLen;
    }
    // 求高度
    public int height(TreeNode root){
        if(root==null){
            return 0;
        }
        int leftheight = height(root.left);
        int rightheight = height(root.right);
        maxLen = Math.max(maxLen,leftheight+rightheight);
        
        return Math.max(leftheight,rightheight)+1;
    }
}
```



### 【22.Leetcode_剑指Offer55 -I】 二叉树的深度

输入一棵二叉树的根节点，求该树的深度。从根节点到叶节点依次经过的节点（含根、叶节点）形成树的一条路径，最长路径的长度为树的深度。

例如：

给定二叉树 [3,9,20,null,null,15,7]，

3

   / \
  9  20
    /  \
   15   7
返回它的最大深度 3 。

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public int maxDepth(TreeNode root) {
        if(root==null){
            return 0;
        }
        int leftheight = maxDepth(root.left);
        int rightheight = maxDepth(root.right);
        return Math.max(leftheight,rightheight)+1;
    }
}
```

### 【23.Leetcode114】 二叉树展开为链表

给你二叉树的根结点 root ，请你将它展开为一个单链表：

展开后的单链表应该同样使用 TreeNode ，其中 right 子指针指向链表中下一个结点，而左子指针始终为 null 。
展开后的单链表应该与二叉树 先序遍历 顺序相同。


示例 1：

输入：root = [1,2,5,3,4,null,6]

![img](https://assets.leetcode.com/uploads/2021/01/14/flaten.jpg)输出：[1,null,2,null,3,null,4,null,5,null,6]

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public void flatten(TreeNode root) {
        if(root==null){
            return;
        }
        // 拉伸
        flatten(root.left);
        flatten(root.right);
        // 记录
        TreeNode left = root.left;
        TreeNode right = root.right;
        // 连接
        root.left = null;
        root.right = left;
        TreeNode cur = root;
        while(cur.right!=null){
            cur = cur.right;
        }
        cur.right = right;
    }
}
```

### 【24.Leetcode96 不同的二叉搜索树】

给你一个整数 n ，求恰由 n 个节点组成且节点值从 1 到 n 互不相同的 二叉搜索树 有多少种？返回满足题意的二叉搜索树的种数。

 

示例 1：

![img](https://assets.leetcode.com/uploads/2021/01/18/uniquebstn3.jpg)


输入：n = 3
输出：5
示例 2：

输入：n = 1
输出：1

```java
class Solution {
    public int numTrees(int n) {
        long  C = 1;
        for(int i=0;i<n;i++){
            C = C * 2 * (2*i+1)/(i+2);
        }
        return (int)C;
    }
}
```

### 【25.Leetcode108 将有序数组转换为二叉搜索树】

给你一个整数数组 nums ，其中元素已经按 升序 排列，请你将其转换为一棵 高度平衡 二叉搜索树。

高度平衡 二叉树是一棵满足「每个节点的左右两个子树的高度差的绝对值不超过 1 」的二叉树。

 

示例 1：

![img](https://assets.leetcode.com/uploads/2021/02/18/btree1.jpg)


输入：nums = [-10,-3,0,5,9]
输出：[0,-3,9,-10,null,5]
解释：[0,-10,5,null,-3,null,9] 也将被视为正确答案

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public TreeNode sortedArrayToBST(int[] nums) {
        return myTree(nums,0,nums.length-1);
    }
    // 构建二叉搜索树
    public TreeNode myTree(int[] nums,int start,int end){
        if(start>end){
            return null;
        }
        int mid = start + ((end-start)>>1);
        TreeNode root = new TreeNode(nums[mid]);
        root.left = myTree(nums,start,mid-1);
        root.right = myTree(nums,mid+1,end);
        return root;
    }
}
```

### 【26.Leetcode113 路径总和II】

给你二叉树的根节点 root 和一个整数目标和 targetSum ，找出所有 从根节点到叶子节点 路径总和等于给定目标和的路径。

叶子节点 是指没有子节点的节点。

 

示例 1：

![img](https://assets.leetcode.com/uploads/2021/01/18/pathsumii1.jpg)


输入：root = [5,4,8,11,null,13,4,7,2,null,null,5,1], targetSum = 22
输出：[[5,4,11,2],[5,8,4,5]]

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    // 结果存储
    List<List<Integer>> res = new ArrayList<>();
    List<Integer> path = new ArrayList<>();
    public List<List<Integer>> pathSum(TreeNode root, int targetSum) {
        // dfs
        dfs(root,targetSum);
        return res;
    }
    // 开始
    public void dfs(TreeNode root,int targetSum){
        if(root==null){
            return ;
        }
        path.add(root.val);
        targetSum -= root.val;

        if(root.left==null&&root.right==null&&targetSum==0){
            res.add(new ArrayList<>(path));
        }
        dfs(root.left,targetSum);
        dfs(root.right,targetSum);

        targetSum += root.val;
        path.remove(path.size()-1);


    }
}
```

### 【27.Leetcode617 合并二叉树】

给定两个二叉树，想象当你将它们中的一个覆盖到另一个上时，两个二叉树的一些节点便会重叠。

你需要将他们合并为一个新的二叉树。合并的规则是如果两个节点重叠，那么将他们的值相加作为节点合并后的新值，否则不为 NULL 的节点将直接作为新二叉树的节点。

示例 1:

输入: 
	Tree 1                     Tree 2                  
          1                         2                             
         / \                       / \                            
        3   2                     1   3                        
       /                           \   \                      
      5                             4   7                  
输出: 
合并后的树:
	     3
	    / \
	   4   5
	  / \   \ 
	 5   4   7

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public TreeNode mergeTrees(TreeNode root1, TreeNode root2) {
        if(root1==null&&root2==null){
            return null;
        }
        if(root1==null){
            return root2;
        }
        if(root2==null){
            return root1;
        }

        TreeNode merge = new TreeNode(root1.val+root2.val);
        merge.left = mergeTrees(root1.left,root2.left);
        merge.right = mergeTrees(root1.right,root2.right);
        return merge; 
    }
}
```

### 【28.Leetcode129 求根节点到叶节点数字之和】

给你一个二叉树的根节点 root ，树中每个节点都存放有一个 0 到 9 之间的数字。
每条从根节点到叶节点的路径都代表一个数字：

例如，从根节点到叶节点的路径 1 -> 2 -> 3 表示数字 123 。
计算从根节点到叶节点生成的 所有数字之和 。

叶节点 是指没有子节点的节点。

 

示例 1：

![img](https://assets.leetcode.com/uploads/2021/02/19/num1tree.jpg)


输入：root = [1,2,3]
输出：25
解释：
从根到叶子节点路径 1->2 代表数字 12
从根到叶子节点路径 1->3 代表数字 13
因此，数字总和 = 12 + 13 = 25

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public int sumNumbers(TreeNode root) {
        // 层序遍历来解题
        int res = 0;
        Queue<TreeNode> queue_node = new LinkedList<>();
        Queue<Integer>  queue_num = new LinkedList<>();
        queue_node.offer(root);
        queue_num.offer(root.val);
        //开始
        while(!queue_node.isEmpty()){
            root = queue_node.poll();
            int temp = queue_num.poll();

            // 根节点了
            if(root.left==null&&root.right==null){
                res += temp;
            }

            if(root.left!=null){
                queue_node.offer(root.left);
                queue_num.offer(temp*10+root.left.val);
            }

            if(root.right!=null){
                queue_node.offer(root.right);
                queue_num.offer(temp*10+root.right.val);
            }

        }
        return res;
    }
}
```

### 【29.Leetcode404 左叶子之和】


计算给定二叉树的所有左叶子之和。

**示例：**

```
    3
   / \
  9  20
    /  \
   15   7

在这个二叉树中，有两个左叶子，分别是 9 和 15，所以返回 24
```

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    int sum = 0;
    public int sumOfLeftLeaves(TreeNode root) {
        // 左叶子之和
        if(root==null){
            return 0;
        }
        if(root.left!=null&&isLeaf(root.left)){
            sum += root.left.val;
        }
        sumOfLeftLeaves(root.left);
        sumOfLeftLeaves(root.right);
        return sum;
    }
    // 判断是否是叶子
    public boolean isLeaf(TreeNode root){
        if(root.left==null&&root.right==null){
            return true;
        }
        return false;
    }
}
```

### [30.Leetcode95 不同的二叉搜索树II](https://leetcode-cn.com/problems/unique-binary-search-trees-ii/)

给你一个整数 n ，请你生成并返回所有由 n 个节点组成且节点值从 1 到 n 互不相同的不同 二叉搜索树 。可以按 任意顺序 返回答案。

 

示例 1：

![img](https://assets.leetcode.com/uploads/2021/01/18/uniquebstn3.jpg)


输入：n = 3
输出：[[1,null,2,null,3],[1,null,3,2],[2,1,3],[3,1,null,null,2],[3,2,null,1]]
示例 2：

输入：n = 1
输出：[[1]]

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public List<TreeNode> generateTrees(int n) {
        if(n==0){
            return new LinkedList<TreeNode>();
        }
        return generateTrees(1,n);
    }

    public List<TreeNode> generateTrees(int start,int end){
        List<TreeNode> allTrees = new LinkedList<>();
        // 递归截止条件
        if(start>end){
            allTrees.add(null);
            return allTrees;
        }
        // 枚举
        for(int i=start;i<=end;i++){
            List<TreeNode> leftTress = generateTrees(start,i-1);
            List<TreeNode> rightTrees = generateTrees(i+1,end);

            for(TreeNode left:leftTress){
                for(TreeNode right:rightTrees){
                    TreeNode curTree = new TreeNode(i);
                    curTree.left = left;
                    curTree.right = right;
                    allTrees.add(curTree);
                }
            }
        }
        return allTrees;
    }
}
```

### [31.Leetcode验证二叉树的前序序列化](https://leetcode-cn.com/problems/verify-preorder-serialization-of-a-binary-tree/)

序列化二叉树的一种方法是使用前序遍历。当我们遇到一个非空节点时，我们可以记录下这个节点的值。如果它是一个空节点，我们可以使用一个标记值记录，例如 #。

     _9_
    /   \
   3     2
  / \   / \
 4   1  #  6
/ \ / \   / \

例如，上面的二叉树可以被序列化为字符串 "9,3,4,#,#,1,#,#,2,#,6,#,#"，其中 # 代表一个空节点。

给定一串以逗号分隔的序列，验证它是否是正确的二叉树的前序序列化。编写一个在不重构树的条件下的可行算法。

每个以逗号分隔的字符或为一个整数或为一个表示 null 指针的 '#' 。

你可以认为输入格式总是有效的，例如它永远不会包含两个连续的逗号，比如 "1,,3" 。

示例 1:

输入: "9,3,4,#,#,1,#,#,2,#,6,#,#"
输出: true
示例 2:

输入: "1,#"
输出: false
示例 3:

输入: "9,#,#,1"
输出: false

```java
class Solution {
    public boolean isValidSerialization(String preorder) {
        // 记录叶子结点和总结点数
        int leftCount = 0, nodeCount = 1;
        // 开始统计
        for(char ch:preorder.toCharArray()){
            //提前截止
            if(leftCount>nodeCount-leftCount){
                return false;
            }

            if(ch==','){
                nodeCount++;
            }
            if(ch=='#'){
                leftCount++;
            }
        }
        // 判断
        return (nodeCount-leftCount)+1 == leftCount;
    }
}
```

### [32.Leetcode341 扁平化嵌套列表迭代器](https://leetcode-cn.com/problems/flatten-nested-list-iterator/)


给你一个嵌套的整型列表。请你设计一个迭代器，使其能够遍历这个整型列表中的所有整数。

列表中的每一项或者为一个整数，或者是另一个列表。其中列表的元素也可能是整数或是其他列表。

 

**示例 1:**

```
输入: [[1,1],2,[1,1]]
输出: [1,1,2,1,1]
解释: 通过重复调用 next 直到 hasNext 返回 false，next 返回的元素的顺序应该是: [1,1,2,1,1]。
```

**示例 2:**

```
输入: [1,[4,[6]]]
输出: [1,4,6]
解释: 通过重复调用 next 直到 hasNext 返回 false，next 返回的元素的顺序应该是: [1,4,6]。
```

```java
/**
 * // This is the interface that allows for creating nested lists.
 * // You should not implement it, or speculate about its implementation
 * public interface NestedInteger {
 *
 *     // @return true if this NestedInteger holds a single integer, rather than a nested list.
 *     public boolean isInteger();
 *
 *     // @return the single integer that this NestedInteger holds, if it holds a single integer
 *     // Return null if this NestedInteger holds a nested list
 *     public Integer getInteger();
 *
 *     // @return the nested list that this NestedInteger holds, if it holds a nested list
 *     // Return empty list if this NestedInteger holds a single integer
 *     public List<NestedInteger> getList();
 * }
 */
public class NestedIterator implements Iterator<Integer> {
    private List<Integer> list = new ArrayList<>();
    private int index = 0;

    public NestedIterator(List<NestedInteger> nestedList) {
        add(nestedList);
    }
    // 添加的函数
    public void add(List<NestedInteger> nestedList){
        // 对其遍历
        for(NestedInteger nestedInteger:nestedList){
            if(nestedInteger.isInteger()){
                list.add(nestedInteger.getInteger());
            }else{
                add(nestedInteger.getList());
            }
        }

    }


    @Override
    public Integer next() {
        return list.get(index++);
    }

    @Override
    public boolean hasNext() {
        return index < list.size();
    }
}

/**
 * Your NestedIterator object will be instantiated and called as such:
 * NestedIterator i = new NestedIterator(nestedList);
 * while (i.hasNext()) v[f()] = i.next();
 */
```

### [33.Leetcode437 路径总和III](https://leetcode-cn.com/problems/path-sum-iii/)

给定一个二叉树，它的每个结点都存放着一个整数值。

找出路径和等于给定数值的路径总数。

路径不需要从根节点开始，也不需要在叶子节点结束，但是路径方向必须是向下的（只能从父节点到子节点）。

二叉树不超过1000个节点，且节点数值范围是 [-1000000,1000000] 的整数。

示例：

root = [10,5,-3,3,2,null,11,3,-2,null,1], sum = 8

      10
     /  \
    5   -3
   / \    \
  3   2   11
 / \   \
3  -2   1

返回 3。和等于 8 的路径有:

1.  5 -> 3
2.  5 -> 2 -> 1
3.  -3 -> 11

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    int sum = 0;
    public int pathSum(TreeNode root, int targetSum) {
        if(root==null){
            return sum;
        }
        // 根左右
        dfs(root,targetSum);
        pathSum(root.left,targetSum);
        pathSum(root.right,targetSum);
        return sum;
    }

    // 判断是否有
    public void dfs(TreeNode root,int targetSum){
        if(root==null){
            return;
        }
        // 前序遍历
        targetSum -= root.val;
        if(targetSum==0){
            sum++;
        }

        dfs(root.left,targetSum);
        dfs(root.right,targetSum);
    }
}
```

### [34.Leetcode450 删除二叉搜索树中的节点](https://leetcode-cn.com/problems/delete-node-in-a-bst/)

给定一个二叉搜索树的根节点 root 和一个值 key，删除二叉搜索树中的 key 对应的节点，并保证二叉搜索树的性质不变。返回二叉搜索树（有可能被更新）的根节点的引用。

一般来说，删除节点可分为两个步骤：

首先找到需要删除的节点；
如果找到了，删除它。
说明： 要求算法时间复杂度为 O(h)，h 为树的高度。

示例:

root = [5,3,6,2,4,null,7]
key = 3

    5
   / \
  3   6
 / \   \
2   4   7

给定需要删除的节点值是 3，所以我们首先找到 3 这个节点，然后删除它。

一个正确的答案是 [5,4,6,2,null,null,7], 如下图所示。

    5
   / \
  4   6
 /     \
2       7

另一个正确答案是 [5,2,6,null,4,null,7]。

    5
   / \
  2   6
   \   \
    4   7

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    // 删除二叉搜索树中的结点

    // 找后继
    public int successor(TreeNode root){
        root = root.right;
        while(root.left!=null){
            root = root.left;
        }
        return root.val;
    }
    // 找前驱
    public int processor(TreeNode root){
        root = root.left;
        while(root.right!=null){
            root = root.right;
        }
        return root.val;
    }
    public TreeNode deleteNode(TreeNode root, int key) {
        // 递归
        if(root==null){
            return null;
        }

        // 查找
        if(root.val>key){
            root.left = deleteNode(root.left,key);
        }else if(root.val<key){
            root.right = deleteNode(root.right,key);
        }else{
            // 删除孩子节点
            if(root.left==null&&root.right==null){
                root = null;
            }
            else if(root.right!=null){
                // 替换
                root.val = successor(root);
                // 删除
                root.right = deleteNode(root.right,root.val);
            }
            else if(root.left!=null){
                root.val = processor(root);
                root.left = deleteNode(root.left,root.val);
            }
        }
        return root;
    }
}
```

### [35.Leetcode515 在每个树行中找最大值](https://leetcode-cn.com/problems/find-largest-value-in-each-tree-row/)

您需要在二叉树的每一行中找到最大的值。

示例：

输入: 

          1
         / \
        3   2
       / \   \  
      5   3   9 

输出: [1, 3, 9]

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public List<Integer> largestValues(TreeNode root) {
        // 层序遍历
        List<Integer> res = new ArrayList<>();
        if(root==null){
            return res;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        // 记录
        int maxValue = Integer.MIN_VALUE;
        while(!queue.isEmpty()){
            int levelSize = queue.size();
            // 开始
            for(int i=0;i<levelSize;i++){
                root = queue.poll();
                maxValue = Math.max(maxValue,root.val);
                if(root.left!=null){
                    queue.offer(root.left);
                }
                if(root.right!=null){
                    queue.offer(root.right);
                }
            }
            res.add(maxValue);
            maxValue = Integer.MIN_VALUE;
        }
        return res;
    }
}
```

### [36.Leetcode637 二叉树的层平均值](https://leetcode-cn.com/problems/average-of-levels-in-binary-tree/)

给定一个非空二叉树, 返回一个由每层节点平均值组成的数组。

 

示例 1：

输入：
    3
   / \
  9  20
    /  \
   15   7
输出：[3, 14.5, 11]
解释：
第 0 层的平均值是 3 ,  第1层是 14.5 , 第2层是 11 。因此返回 [3, 14.5, 11] 。

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public List<Double> averageOfLevels(TreeNode root) {
        // 层序遍历
        List<Double> res = new ArrayList<>();
        if(root==null){
            return res;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        double avg = 0;
        while(!queue.isEmpty()){
            int levelSize = queue.size();
            
            for(int i=0;i<levelSize;i++){
                root = queue.poll();
                avg += root.val;
                if(root.left!=null){
                    queue.offer(root.left);
                }
                if(root.right!=null){
                    queue.offer(root.right);
                }
            }
            // 计算
            res.add(avg/levelSize);
            avg = 0;
        }
        return res;
    }
}
```

