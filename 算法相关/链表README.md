# 链表篇

### 22.反转链表

#### 22.1 反转链表(递归和迭代法)

```java
public class Listnode{
    int val;
    ListNode next;
    ListNode pre;
    ListNode (int val){
        this.val = val;
    }
}
//递归
public class Solution{
    public ListNode revoseList(ListNode head){
        if(head == null || head.next == null){
            return head;
        }
        ListNode cur = revoseList(head.next);
        head.next.next = head;
        head.next = null;
        return cur;
    }
}
```

```java
public class Solution{
    public ListNode revoseList(ListNode head){
        if(head == null || head.next == null){
            return head;
        }
        ListNode pre = null;
        ListNode cur = head;
        ListNode temp = null;
        while(cur != null){
            //标记下一个节点，以免丢失
            temp = cur.next;
            //实现反转
            cur.next = pre;
            pre = cur;
            cur = temp;
        }
        return pre;
    }
}
```

#### 22.2 K个一组反转链表

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode reverseKGroup(ListNode head, int k) {
        //定义一个头指针，用来将后面所有节点统一处理
        ListNode dum = new ListNode();
        dum.next = head;
        //用来标记每一次反转部分的前一个结点
        ListNode pre = dum;
        //运动指针，扫描要反转的部分
        ListNode end = dum;

        while(end.next != null){
            //每次扫描完要反转的部分，如果end为空说明达到尾部，直接break
            for(int i = 0; i < k && end != null;i++) end = end.next;
            if(end == null) break;
            //标记要反转部分的开头节点
            ListNode start = pre.next;
            //标记反转链表后面部分的第一个节点，避免丢失
            ListNode next = end.next;
            //将要反转部分向后的指针断开，准备反转
            end.next = null;
            //反转完的链表返回反转后的头结点，接到pre的后面
            pre.next = reverse(start);
            //反转后start指针应该指向反转完成部分的末尾
            start.next = next;
            //让pre和end指针继续指向待反转链表的前一个节点
            pre = start;
            end = pre;
        }
        return dum.next;
    }
    public ListNode reverse(ListNode start){
        ListNode cur = start;
        ListNode pre1 = null;
        while(cur != null){
            ListNode temp = cur.next;
            cur.next = pre1;
            pre1 = cur;
            cur = temp;
        }
        return pre1;
    }
}

```

### 

#### Leetcode链表翻转题

##### Leetcode206 反转链表

给你单链表的头节点 head ，请你反转链表，并返回反转后的链表。

示例 1：

![img](https://assets.leetcode.com/uploads/2021/02/19/rev1ex1.jpg)


输入：head = [1,2,3,4,5]
输出：[5,4,3,2,1]

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode reverseList(ListNode head) {
        ListNode pre = null;
        ListNode next = null;
        while(head!=null){
            next = head.next;
            head.next = pre;
            pre = head;
            head = next;
        }
        return pre;
    }
}
```

##### Leetcode092 反转链表II

给你单链表的头指针 head 和两个整数 left 和 right ，其中 left <= right 。请你反转从位置 left 到位置 right 的链表节点，返回 反转后的链表 。

示例 1：

![img](https://assets.leetcode.com/uploads/2021/02/19/rev2ex2.jpg)

输入：head = [1,2,3,4,5], left = 2, right = 4
输出：[1,4,3,2,5]
示例 2：

输入：head = [5], left = 1, right = 1
输出：[5]

##### Leetcode025 K个一组翻转链表

给你一个链表，每 k 个节点一组进行翻转，请你返回翻转后的链表。

k 是一个正整数，它的值小于或等于链表的长度。

如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序。

进阶：

你可以设计一个只使用常数额外空间的算法来解决此问题吗？
你不能只是单纯的改变节点内部的值，而是需要实际进行节点交换。

示例 1：

![img](https://assets.leetcode.com/uploads/2020/10/03/reverse_ex1.jpg)


输入：head = [1,2,3,4,5], k = 2
输出：[2,1,4,3,5]

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode reverseKGroup(ListNode head, int k) {
        ListNode firstHead = head;
        ListNode firstTail = head;
        for(int i=0;i<k;i++){
            if(firstTail==null){
                return head;
            }
            firstTail = firstTail.next;
        }
        ListNode newHead = reverse(firstHead,firstTail);
        firstHead.next = reverseKGroup(firstTail,k);

        return newHead;
    }

    public ListNode reverse(ListNode head,ListNode tail){
        ListNode pre = null;
        ListNode next = null;
        while(head!=tail){
            next = head.next;
            head.next = pre;
            pre = head;
            head = next;
        }
        return pre;
    }
}
```

##### Leetcode024 两两交换链表中的节点

给定一个链表，两两交换其中相邻的节点，并返回交换后的链表。

你不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。

 

示例 1：

![img](https://assets.leetcode.com/uploads/2020/10/03/swap_ex1.jpg)

输入：head = [1,2,3,4]
输出：[2,1,4,3]

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode swapPairs(ListNode head) {
        // 两两交换
        ListNode dummy = new ListNode(-1,head);
        ListNode node = dummy;
        while(node.next!=null&&node.next.next!=null){
            ListNode first = node.next;
            ListNode second = node.next.next;

            first.next = second.next;
            node.next = second;
            second.next = first;

            node = first;
        }
        return dummy.next;
    }
}
```

##### Leetcode328 奇偶链表

给定一个单链表，把所有的奇数节点和偶数节点分别排在一起。请注意，这里的奇数节点和偶数节点指的是节点编号的奇偶性，而不是节点的值的奇偶性。

请尝试使用原地算法完成。你的算法的空间复杂度应为 O(1)，时间复杂度应为 O(nodes)，nodes 为节点总数。

示例 1:

输入: 1->2->3->4->5->NULL
输出: 1->3->5->2->4->NULL
示例 2:

输入: 2->1->3->5->6->4->7->NULL 
输出: 2->3->6->7->1->5->4->NULL

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode oddEvenList(ListNode head) {
        ListNode oddDummy = new ListNode();
        ListNode oddTail = oddDummy;
        ListNode evenDummy = new ListNode();
        ListNode evenTail = evenDummy;
        //遍历原链表
        boolean isOdd = true;
        while(head!=null){
            if(isOdd){
                oddTail.next = head;
                oddTail = oddTail.next;
            }else{
                evenTail.next = head;
                evenTail = evenTail.next;
            }
            head = head.next;
            isOdd = !isOdd;
        }
        evenTail.next = null;
        oddTail.next = evenDummy.next;
        return oddDummy.next;
    }
}
```



```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode oddEvenList(ListNode head) {
        if(head==null || head.next==null){
            return head;
        }
        ListNode oddDummy = head;
        ListNode oddHead  = oddDummy;
        ListNode evenDummy = head.next;
        ListNode evenHead = evenDummy;
        // 继续
        while(oddHead.next!=null&&evenHead.next!=null){
            oddHead.next = evenHead.next;
            oddHead = oddHead.next;

            evenHead.next = oddHead.next;
            evenHead = evenHead.next;
        }
        // 连接
        oddHead.next = evenDummy;
        return oddDummy;
    }
}
```



##### Leetcode086 分割链表

给你一个链表的头节点 head 和一个特定值 x ，请你对链表进行分隔，使得所有 小于 x 的节点都出现在 大于或等于 x 的节点之前。

你应当 保留 两个分区中每个节点的初始相对位置。

 

示例 1：

![img](https://assets.leetcode.com/uploads/2021/01/04/partition.jpg)


输入：head = [1,4,3,2,5,2], x = 3
输出：[1,2,2,4,3,5]

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode partition(ListNode head, int x) {
        ListNode smallDummy = new ListNode(-1,head);
        ListNode bigDummy = new ListNode(-1,head);
        ListNode smallHead = smallDummy;
        ListNode bigHead = bigDummy;
        while(head!=null){
            if(head.val<x){
                smallHead.next = head;
                smallHead = smallHead.next;
            }else{
                bigHead.next = head;
                bigHead = bigHead.next;
            }
            head = head.next;
        }
        bigHead.next = null;
        smallHead.next = bigDummy.next;
        return smallDummy.next;
    }
}
```

##### Leetcode143 重排链表

给定一个单链表 L：L0→L1→…→Ln-1→Ln ，
将其重新排列后变为： L0→Ln→L1→Ln-1→L2→Ln-2→…

你不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。

示例 1:

给定链表 1->2->3->4, 重新排列为 1->4->2->3.
示例 2:

给定链表 1->2->3->4->5, 重新排列为 1->5->2->4->3.

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public void reorderList(ListNode head) {
        //重排链表
        ListNode mid = findMedium(head);
        // 记录
        ListNode newHead = mid.next;
        mid.next = null;
        //翻转链表
        newHead = reverse(newHead);
        // 合并这两个链表
        merge(head,newHead);
    }
    public void merge(ListNode l1,ListNode l2){
        while(l1!=null&&l2!=null){
            ListNode next1 = l1.next;
            ListNode next2 = l2.next;
            //开始
            l1.next = l2;
            l2.next = next1;
            //继续
            l1 = next1;
            l2 = next2;
        }

    }

    public ListNode findMedium(ListNode head){
        ListNode slow = head;
        ListNode fast = head;
        while(fast!=null&&fast.next!=null){
            slow = slow.next;
            fast = fast.next.next;
        }
        return slow;
    }

    public ListNode reverse(ListNode head){
        ListNode pre = null;
        ListNode next = null;
        while(head!=null){
            next = head.next;
            head.next = pre;
            pre = head;
            head = next;
        }
        return pre;
    }
}
```

#### Leetcode排序链表删除重复元素

##### Leetcode083 删除排序链表中的重复元素

存在一个按升序排列的链表，给你这个链表的头节点 head ，请你删除所有重复的元素，使每个元素 只出现一次 。

返回同样按升序排列的结果链表。

 

示例 1：

![img](https://assets.leetcode.com/uploads/2021/01/04/list1.jpg)


输入：head = [1,1,2]
输出：[1,2]

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode deleteDuplicates(ListNode head) {
        ListNode l = head;
        while(l!=null&&l.next!=null){
            if(l.val==l.next.val){
                l.next = l.next.next;
            }else{
                l = l.next;
            }
        }
        return head;
    }
}
```

##### Leetcode082 删除排序链表中的重复元素II

存在一个按升序排列的链表，给你这个链表的头节点 head ，请你删除链表中所有存在数字重复情况的节点，只保留原始链表中 没有重复出现 的数字。

返回同样按升序排列的结果链表。

 

示例 1：

![img](https://assets.leetcode.com/uploads/2021/01/04/linkedlist1.jpg)


输入：head = [1,2,3,3,4,4,5]
输出：[1,2,5]

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode deleteDuplicates(ListNode head) {
        ListNode dummy = new ListNode(-1,head);
        ListNode l = dummy;
        // 判断是否有重复节点
        while(l.next!=null&&l.next.next!=null){
            if(l.next.val==l.next.next.val){
                int val = l.next.val;
                while(l.next!=null&&l.next.val==val){
                    l.next = l.next.next;
                }
            }else{
                l = l.next;
            }
        }
        return dummy.next;
    }
}
```

#### 

### 1.[Leetcode002 两数相加](https://leetcode-cn.com/problems/add-two-numbers/)

给你两个 非空 的链表，表示两个非负的整数。它们每位数字都是按照 逆序 的方式存储的，并且每个节点只能存储 一位 数字。

请你将两个数相加，并以相同形式返回一个表示和的链表。

你可以假设除了数字 0 之外，这两个数都不会以 0 开头。

 

示例 1：

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2021/01/02/addtwonumber1.jpg)


输入：l1 = [2,4,3], l2 = [5,6,4]
输出：[7,0,8]
解释：342 + 465 = 807.

> 链表的：加法模板
>
> 记录好进位值

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        // 链表的两数相加的加法模板
        int rem = 0;
        // 结果
        ListNode dummy = new ListNode(-1);
        ListNode l = dummy;
        // 开始
        while(l1!=null || l2!=null){
            int t1 = l1!=null?l1.val:0;
            int t2 = l2!=null?l2.val:0;
            // 相加
            int temp = t1+t2+rem;
            rem = temp/10;
            l.next = new ListNode(temp%10);
            // 开始继续走
            l = l.next;
            if(l1!=null){
                l1 = l1.next;
            }
            if(l2!=null){
                l2 = l2.next;
            }
        }
        // 如果还有剩余
        if(rem!=0){
            l.next = new ListNode(rem);
        }
        return dummy.next;
    }
}
```

### 2.[Leetcode019删除链表的倒数第N个结点](https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/)

给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。

进阶：你能尝试使用一趟扫描实现吗？

 

示例 1：

![img](https://assets.leetcode.com/uploads/2020/10/03/remove_ex1.jpg)


输入：head = [1,2,3,4,5], n = 2
输出：[1,2,3,5]

> 找倒数第n+1个

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode removeNthFromEnd(ListNode head, int n) {
        // 删除倒数第n个，找个倒数第n+1个
        ListNode dummy = new ListNode(-1,head);
        ListNode slow = dummy;
        ListNode fast = dummy;
        for(int i=0;i<n;i++){
            fast = fast.next;
        }
        while(fast.next!=null){
            slow = slow.next;
            fast = fast.next;
        }
        slow.next = slow.next.next;
        return dummy.next;
    }
}
```

### [3.Leetcode021合并两个有序链表](https://leetcode-cn.com/problems/merge-two-sorted-lists/)

将两个升序链表合并为一个新的 升序 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。 

 

示例 1：

![img](https://assets.leetcode.com/uploads/2020/10/03/merge_ex1.jpg)


输入：l1 = [1,2,4], l2 = [1,3,4]
输出：[1,1,2,3,4,4]

> 合并链表，很简单的

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        // 合并的新链表
        ListNode dummy = new ListNode(-1);
        ListNode l = dummy;
        while(l1!=null&&l2!=null){
            if(l1.val>l2.val){
                l.next = l2;
                l = l.next;
                l2 = l2.next;
            }else{
                l.next = l1;
                l = l.next;
                l1 = l1.next;
            }
        }
        if(l1!=null){
            l.next = l1;
        }
        if(l2!=null){
            l.next = l2;
        }
        return dummy.next;
    }
}
```

### [4.Leetcode023 合并K个升序链表](https://leetcode-cn.com/problems/merge-k-sorted-lists/)

给你一个链表数组，每个链表都已经按升序排列。

请你将所有链表合并到一个升序链表中，返回合并后的链表。

 

示例 1：

输入：lists = [[1,4,5],[1,3,4],[2,6]]
输出：[1,1,2,3,4,4,5,6]
解释：链表数组如下：
[
  1->4->5,
  1->3->4,
  2->6
]
将它们合并到一个有序链表中得到。
1->1->2->3->4->4->5->6

> 解题思路：优先级队列来辅助解题

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        // 合并K个升序链表
        PriorityQueue<ListNode> queue = new PriorityQueue<>((a,b)->(a.val-b.val));
        // 先存放
        for(int i=0;i<lists.length;i++){
            if(lists[i]!=null){
                queue.offer(lists[i]);
            }
        }
        // 继续
        // 结果存储
        ListNode dummy = new ListNode(-1);
        ListNode head = dummy;
        while(!queue.isEmpty()){
            ListNode node = queue.poll();
            head.next = new ListNode(node.val);
            head = head.next;
            if(node.next!=null){
                queue.offer(node.next);
            }
        }
        return dummy.next;
    }
}
```

### [5.Leetcode024 两两交换链表中的节点](https://leetcode-cn.com/problems/swap-nodes-in-pairs/)

给定一个链表，两两交换其中相邻的节点，并返回交换后的链表。

你不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。

 

示例 1：

![img](https://assets.leetcode.com/uploads/2020/10/03/swap_ex1.jpg)


输入：head = [1,2,3,4]
输出：[2,1,4,3]

> 解题思路：两两交换重点，在于开一个哑结点，之后内部一开始first.next=second.next

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode swapPairs(ListNode head) {
        ListNode dummy = new ListNode(-1,head);
        ListNode node = dummy;
        while(head!=null&&head.next!=null){
            //新节点
            ListNode first = head;
            ListNode second = head.next;
            // 开始
            first.next = second.next;
            node.next  = second;
            second.next = first;
            // 重新开始新一轮
            node = first;
            head = first.next;
        }
        return dummy.next;
    }
}
```

### [6.Leetcode025 K个一组翻转链表](https://leetcode-cn.com/problems/reverse-nodes-in-k-group/)

给你一个链表，每 k 个节点一组进行翻转，请你返回翻转后的链表。

k 是一个正整数，它的值小于或等于链表的长度。

如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序。

进阶：

你可以设计一个只使用常数额外空间的算法来解决此问题吗？
你不能只是单纯的改变节点内部的值，而是需要实际进行节点交换。

示例 1：

![img](https://assets.leetcode.com/uploads/2020/10/03/reverse_ex1.jpg)


输入：head = [1,2,3,4,5], k = 2
输出：[2,1,4,3,5]

> 解题思路：递归的思路
>
> 先来翻转第一次的k个，翻转完之后。
>
> 最重要要递归，调用第一次的
>
> first.next= reverseKGroup(second,k);

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode reverseKGroup(ListNode head, int k) {
        // 递归截止条件
        if(head==null){
            return head;
        }
        // K个一组翻转
        ListNode first = head;
        ListNode second = head;
        for(int i=0;i<k;i++){
            if(second==null){
                return head;
            }
            second = second.next;
        }
        // 开始翻转
        ListNode newHead = reverse(first,second);
        // 连接上
        first.next  = reverseKGroup(second,k);
        return newHead;
    }

    // 反转链表
    public ListNode reverse(ListNode head,ListNode tail){
        ListNode pre = null;
        ListNode next = null;
        while(head!=tail){
            next = head.next;
            head.next = pre;
            pre = head;
            head = next;
        }
        return pre;
    }
}
```

### [7.Leetcode061 旋转链表](https://leetcode-cn.com/problems/rotate-list/)

给你一个链表的头节点 head ，旋转链表，将链表每个节点向右移动 k 个位置。

 

示例 1：

![img](https://assets.leetcode.com/uploads/2020/11/13/rotate1.jpg)


输入：head = [1,2,3,4,5], k = 2
输出：[4,5,1,2,3]

> 分几步走

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode rotateRight(ListNode head, int k) {
        // 向右移动k个位置
        // 解题思路：先连接起来
        if(head==null){
            return head;
        }
        int len = 1;
        ListNode tail = head;
        while(tail.next!=null){
            len++;
            tail = tail.next;
        }
        // 循环链表
        tail.next = head;
        // 第二步 继续找新链表的尾巴
        k = k%len;
        tail = head;
        for(int i=0;i<len-k-1;i++){
            tail = tail.next;
        }
        head = tail.next;
        // 断开
        tail.next = null;
        return head;

    }
}
```

### [8.Leetcode082 删除排序链表中的重复元素II](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list-ii/)

存在一个按升序排列的链表，给你这个链表的头节点 head ，请你删除链表中所有存在数字重复情况的节点，只保留原始链表中 没有重复出现 的数字。

返回同样按升序排列的结果链表。

 

示例 1：


输入：head = [1,2,3,3,4,4,5]
输出：[1,2,5]

> 删除全部重复元素，
>
> while循环+if判断

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode deleteDuplicates(ListNode head) {
        // 删除只要重复数字的
        ListNode dummy = new ListNode(-1,head);
        ListNode cur = dummy;
       while(cur.next!=null&&cur.next.next!=null){
           // 直接判断即可了
           if(cur.next.val==cur.next.next.val){
               // 相等如何处理
               int x = cur.next.val;
               while(cur.next!=null&&x==cur.next.val){
                   cur.next = cur.next.next;
               }
           }else{   
               cur = cur.next;
           }
       }
       return dummy.next;
    }
}
```

### 9.Leetcode083 删除排序链表中的重复元素

存在一个按升序排列的链表，给你这个链表的头节点 head ，请你删除所有重复的元素，使每个元素 只出现一次 。

返回同样按升序排列的结果链表。

 

示例 1：

![img](https://assets.leetcode.com/uploads/2021/01/04/list1.jpg)


输入：head = [1,1,2]
输出：[1,2]

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode deleteDuplicates(ListNode head) {
        ListNode cur = head;
        while(cur!=null&&cur.next!=null){
            // 直接判断
            if(cur.val==cur.next.val){
                cur.next = cur.next.next;
            }else{
                cur = cur.next;
            }
        }
        return head;
    }
}
```

### [10.Leetcode086 分割链表](https://leetcode-cn.com/problems/partition-list/)

给你一个链表的头节点 head 和一个特定值 x ，请你对链表进行分隔，使得所有 小于 x 的节点都出现在 大于或等于 x 的节点之前。

你应当 保留 两个分区中每个节点的初始相对位置。

 

示例 1：

![img](https://assets.leetcode.com/uploads/2021/01/04/partition.jpg)


输入：head = [1,4,3,2,5,2], x = 3
输出：[1,2,2,4,3,5]

> 解题思路：两个链表

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    // 分割链表
    public ListNode partition(ListNode head, int x) {
        ListNode smallDummy = new ListNode(-1);
        ListNode smallHead = smallDummy;
        ListNode bigDummy = new ListNode(-1);
        ListNode bigHead  = bigDummy;
        // 开始走
        while(head!=null){
            if(head.val>=x){
                bigHead.next = head;
                bigHead = bigHead.next;
            }else{
                smallHead.next = head;
                smallHead = smallHead.next;
            }
            head = head.next;
        }
        // 拼接起来
        bigHead.next = null;
        smallHead.next = bigDummy.next;
        return smallDummy.next;
    }
}
```



### [11.Leetcode092 反转链表II](https://leetcode-cn.com/problems/reverse-linked-list-ii/)

给你单链表的头指针 head 和两个整数 left 和 right ，其中 left <= right 。请你反转从位置 left 到位置 right 的链表节点，返回 反转后的链表 。

示例 1：

![img](https://assets.leetcode.com/uploads/2021/02/19/rev2ex2.jpg)

输入：head = [1,2,3,4,5], left = 2, right = 4
输出：[1,4,3,2,5]

> 繁琐在于，记录之前的，记录更改之后的

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode reverseBetween(ListNode head, int left, int right) {
        ListNode dummy = new ListNode(-1,head);
        // 开始行走了
        ListNode pre = dummy;
        for(int i=1;i<left;i++){
            pre = pre.next;
        }
        ListNode oldHead = pre.next;
        ListNode newTail = pre.next;
        // 继续走
        for(int i=left;i<right;i++){
            newTail = newTail.next;
        }
        ListNode next = newTail.next;
        // 反转
        ListNode newHead = reverse(oldHead,next);

        //重新连接
        pre.next = newHead;
        oldHead.next = next;
        return dummy.next;
    }

    public ListNode reverse(ListNode head,ListNode tail){
        ListNode next = null;
        ListNode pre  = null;
        while(head!=tail){
            next = head.next;
            head.next = pre;
            pre = head;
            head = next;
        }
        return pre;
    }
}
```

### [12.Leetcode109 有序链表转换二叉搜索树](https://leetcode-cn.com/problems/convert-sorted-list-to-binary-search-tree/)

给定一个单链表，其中的元素按升序排序，将其转换为高度平衡的二叉搜索树。

本题中，一个高度平衡二叉树是指一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过 1。

示例:

给定的有序链表： [-10, -3, 0, 5, 9],

一个可能的答案是：[0, -3, 9, -10, null, 5], 它可以表示下面这个高度平衡二叉搜索树：

      0
     / \
   -3   9
   /   /
 -10  5

> 有序链表转换二叉搜索树

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
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
    // 二叉搜索树
    public TreeNode sortedListToBST(ListNode head) {
        // 建立新的
        return ListToBST(head,null);
    }
    public TreeNode ListToBST(ListNode head,ListNode tail){
        // 递归截止条件
        if(head==tail){
            return null;
        }
        ListNode mid = findMedium(head,tail);
        TreeNode root = new TreeNode(mid.val);
        root.left = ListToBST(head,mid);
        // 重点
        root.right = ListToBST(mid.next,tail);
        return root;
    }

    public ListNode findMedium(ListNode head,ListNode tail){
        ListNode slow = head;
        ListNode fast = head;
        while(fast!=tail&&fast.next!=tail){
            slow = slow.next;
            fast = fast.next.next;
        }
        return slow;
    }
}
```

### [13.Leetcode138 复制带随机指针的链表](https://leetcode-cn.com/problems/copy-list-with-random-pointer/)

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/01/09/e1.png)

输入：head = [[7,null],[13,0],[11,4],[10,2],[1,0]]
输出：[[7,null],[13,0],[11,4],[10,2],[1,0]]

> 解题思路 ：用HashMap

```java
/*
// Definition for a Node.
class Node {
    int val;
    Node next;
    Node random;

    public Node(int val) {
        this.val = val;
        this.next = null;
        this.random = null;
    }
}
*/

class Solution {
    public Node copyRandomList(Node head) {
        // 用HashMap来解题
        HashMap<Node,Node> hashMap = new HashMap<>();
        Node node = head;
        while(node!=null){
            hashMap.put(node,new Node(node.val));
            node = node.next;
        }
        // 开始记录连接
        node = head;
        while(node!=null){
            hashMap.get(node).next = hashMap.get(node.next);
            hashMap.get(node).random = hashMap.get(node.random);
            node = node.next;
        }
        return hashMap.get(head);
    }
}
```

### [14.Leetcode141 环形链表](https://leetcode-cn.com/problems/linked-list-cycle/)

给定一个链表，判断链表中是否有环。

如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，我们使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 pos 是 -1，则在该链表中没有环。注意：pos 不作为参数进行传递，仅仅是为了标识链表的实际情况。

如果链表中存在环，则返回 true 。 否则，返回 false 。

 

进阶：

你能用 O(1)（即，常量）内存解决此问题吗？

 

示例 1：

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/07/circularlinkedlist.png)

输入：head = [3,2,0,-4], pos = 1
输出：true
解释：链表中有一个环，其尾部连接到第二个节点

> 快慢指针解决环形链表问题

```java
/**
 * Definition for singly-linked list.
 * class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) {
 *         val = x;
 *         next = null;
 *     }
 * }
 */
public class Solution {
    public boolean hasCycle(ListNode head) {
        // 快慢指针
        ListNode slow = head;
        ListNode fast = head;
        while(fast!=null&&fast.next!=null){
            slow = slow.next;
            fast = fast.next.next;
            if(slow==fast){
                return true;
            }
        }
        return false;
    }
}
```

### [15.Leetcode142 环形链表II](https://leetcode-cn.com/problems/linked-list-cycle-ii/)


给定一个链表，返回链表开始入环的第一个节点。 如果链表无环，则返回 `null`。

为了表示给定链表中的环，我们使用整数 `pos` 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 `pos` 是 `-1`，则在该链表中没有环。**注意，`pos` 仅仅是用于标识环的情况，并不会作为参数传递到函数中。**

**说明：**不允许修改给定的链表。

**进阶：**

- 你是否可以使用 `O(1)` 空间解决此题？

 

**示例 1：**

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/07/circularlinkedlist.png)

```
输入：head = [3,2,0,-4], pos = 1
输出：返回索引为 1 的链表节点
解释：链表中有一个环，其尾部连接到第二个节点。
```

> 解题思路：快慢指针。

```java
/**
 * Definition for singly-linked list.
 * class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) {
 *         val = x;
 *         next = null;
 *     }
 * }
 */
public class Solution {
    public ListNode detectCycle(ListNode head) {
        ListNode slow = head;
        ListNode fast = head;
        while(fast!=null&&fast.next!=null){
            slow = slow.next;
            fast = fast.next.next;
            if(slow==fast){
                fast = head;
                while(slow!=fast){
                    slow = slow.next;
                    fast = fast.next;
                }
                return slow.val;
            }
        }  
        return null;
    }
}
```

### [16.Leetcode143 重排链表](https://leetcode-cn.com/problems/reorder-list/)

给定一个单链表 L：L0→L1→…→Ln-1→Ln ，
将其重新排列后变为： L0→Ln→L1→Ln-1→L2→Ln-2→…

你不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。

示例 1:

给定链表 1->2->3->4, 重新排列为 1->4->2->3.
示例 2:

给定链表 1->2->3->4->5, 重新排列为 1->5->2->4->3.

> 解题思路：仔细观察排列后的新链表 为 1->2->3 和5->4 然后一个接一个即可了。

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public void reorderList(ListNode head) {
        // 1.中间点找
        ListNode mid = findMedium(head);
        // 2.翻转
        ListNode newHead = reverse(mid.next);
        mid.next = null;
        // 3.最后两个链表合并了
        merge(head,newHead);
    }

    public ListNode findMedium(ListNode head){
        ListNode slow = head;
        ListNode fast = head;
        while(fast!=null&&fast.next!=null){
            slow = slow.next;
            fast = fast.next.next;
        }
        return slow;
    }

    public ListNode reverse(ListNode head){
        ListNode pre = null;
        ListNode next = null;
        while(head!=null){
            next = head.next;
            head.next = pre;
            pre = head;
            head = next;
        }
        return pre;
    }

    public void merge(ListNode a,ListNode b){
        while(a!=null&&b!=null){
            ListNode next1 = a.next;
            ListNode next2 = b.next;
            
            a.next = b;
            b.next = next1;

            a = next1;
            b = next2;
        }

    }
}
```

### [17.Leetcode147 对链表进行插入排序](https://leetcode-cn.com/problems/insertion-sort-list/)

**示例 1：**

```
输入: 4->2->1->3
输出: 1->2->3->4
```

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode insertionSortList(ListNode head) {
        if(head==null || head.next == null) {
	    		return head;
	    	}
	    	// 哑结点
	    	ListNode dummy = new ListNode(-1,head);
	    	// 遍历结束
	    	while(head!=null&&head.next!=null) {
	    		// 先找到没拍好序的
	    		if(head.val<=head.next.val) {
	    			head = head.next;
	    			continue;
	    		}
	    		// 找到要插入的位置
	    		ListNode pre = dummy;
	    		while(pre.next.val<head.next.val) {
	    			pre = pre.next;
	    		}
	    		ListNode cur = head.next;
	    		//
	    		head.next = cur.next;
	    		cur.next = pre.next;
	    		pre.next = cur;	    		
	    	}
	    	return dummy.next;
    }
}


```

### [18.Leetcode206  反转链表](https://leetcode-cn.com/problems/reverse-linked-list/)

给你单链表的头节点 head ，请你反转链表，并返回反转后的链表。

示例 1：

![img](https://assets.leetcode.com/uploads/2021/02/19/rev1ex1.jpg)


输入：head = [1,2,3,4,5]
输出：[5,4,3,2,1]

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode reverseList(ListNode head) {
        ListNode pre = null;
        ListNode next = null;
        while(head!=null){
            next = head.next;
            head.next = pre;
            pre = head;
            head = next;
        }
        return pre;
    }
}
```

### [19.Leetcode160 相交链表](https://leetcode-cn.com/problems/intersection-of-two-linked-lists/)

编写一个程序，找到两个单链表相交的起始节点。

如下面的两个链表**：**

[![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/14/160_statement.png)](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/14/160_statement.png)

在节点 c1 开始相交。

 

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) {
 *         val = x;
 *         next = null;
 *     }
 * }
 */
public class Solution {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode newHeadA = headA;
        ListNode newHeadB = headB;
        while(headA!=headB){
            headA = headA!=null?headA.next:newHeadB;
            headB = headB!=null?headB.next:newHeadA;
        }
        return headA;
    }
}
```

### [20.Leetcode148 排序链表](https://leetcode-cn.com/problems/sort-list/)

给你链表的头结点 head ，请将其按 升序 排列并返回 排序后的链表 。

进阶：

你可以在 O(n log n) 时间复杂度和常数级空间复杂度下，对链表进行排序吗？

示例 1：

![img](https://assets.leetcode.com/uploads/2020/09/14/sort_list_1.jpg)


输入：head = [4,2,1,3]
输出：[1,2,3,4]

> 排序方式：归并排序
>
> 先递归 递归的过程中得到left的结果和right的结果记得！！！
>
> 后归并

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode sortList(ListNode head) {
        if(head==null||head.next==null){
            return head;
        }
        // 调用归并排序
        return mergeSort(head,null);
    }
    // 归并排序
    public ListNode mergeSort(ListNode head,ListNode tail){
        // 判断
        if(head.next==tail){
            head.next = null;
            return head;
        }
        // 找中点
        ListNode slow = head;
        ListNode fast = head;
        while(fast!=tail&&fast.next!=tail){
            slow = slow.next;
            fast = fast.next.next;
        }
        ListNode mid = slow;
        // 递归
        ListNode left = mergeSort(head,mid);
        ListNode right = mergeSort(mid,tail);
        ListNode sorted = merge(left,right);
        return sorted;
    }

    // 合并
    public ListNode merge(ListNode l1,ListNode l2){
        ListNode dummy = new ListNode(-1);
        ListNode l = dummy;
        while(l1!=null&&l2!=null){
            if(l1.val<=l2.val){
                l.next = l1;
                l = l.next;
                l1 = l1.next;
            }else{
                l.next = l2;
                l = l.next;
                l2 = l2.next;
            }
        }
        if(l1!=null){
            l.next = l1;
        }
        if(l2!=null){
            l.next = l2;
        }
        return dummy.next;

    }
}
```

### [21.Leetcode203 移除链表元素](https://leetcode-cn.com/problems/remove-linked-list-elements/)

给你一个链表的头节点 head 和一个整数 val ，请你删除链表中所有满足 Node.val == val 的节点，并返回 新的头节点 。

示例 1：

![img](https://assets.leetcode.com/uploads/2021/03/06/removelinked-list.jpg)


输入：head = [1,2,6,3,4,5,6], val = 6
输出：[1,2,3,4,5]

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode removeElements(ListNode head, int val) {
        // 添加哑结点
        ListNode dummy = new ListNode(-1,head);
        ListNode node = dummy;
        while(node.next!=null){
            if(node.next.val==val){
                node.next = node.next.next;
            }else{
                node = node.next;
            }
        }
        return dummy.next;
    }
}
```

### 22[Leetcode234 回文链表](https://leetcode-cn.com/problems/palindrome-linked-list/)

请判断一个链表是否为回文链表。

**示例 1:**

```
输入: 1->2
输出: false
```

**示例 2:**

```
输入: 1->2->2->1
输出: true
```

**进阶：**
你能否用 O(n) 时间复杂度和 O(1) 空间复杂度解决此题？

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public boolean isPalindrome(ListNode head) {
        // 回文链表
        // 1.先找中间的点
        ListNode medium = findMedium(head);
        // 2.反转后面的
        ListNode newHead = reverse(medium.next);
        medium.next = null;
        // 3.对比
        while(head!=null&&newHead!=null){
            if(head.val!=newHead.val){
                return false;
            }
            head = head.next;
            newHead = newHead.next;
        }
        return true;
    }
    // 1.找中间的点
    public ListNode findMedium(ListNode head){
        ListNode slow = head;
        ListNode fast = head;
        while(fast.next!=null&&fast.next.next!=null){
            slow = slow.next;
            fast = fast.next.next;
        }
        return slow;
    }
    // 2.反转链表
    public ListNode reverse(ListNode head){
        ListNode pre = null;
        ListNode next = null;
        while(head!=null){
            next = head.next;
            head.next = pre;
            pre = head;
            head = next;
        }
        return pre;
    }
}
```

### [23.Leetcode237 删除链表中的节点](https://leetcode-cn.com/problems/delete-node-in-a-linked-list/)

请编写一个函数，使其可以删除某个链表中给定的（非末尾）节点。传入函数的唯一参数为 要被删除的节点 。

 

现有一个链表 -- head = [4,5,1,9]，它可以表示为:

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2019/01/19/237_example.png)

 

示例 1：

输入：head = [4,5,1,9], node = 5
输出：[4,1,9]
解释：给定你链表中值为 5 的第二个节点，那么在调用了你的函数之后，该链表应变为 4 -> 1 -> 9.
示例 2：

输入：head = [4,5,1,9], node = 1
输出：[4,5,9]
解释：给定你链表中值为 1 的第三个节点，那么在调用了你的函数之后，该链表应变为 4 -> 5 -> 9.

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public void deleteNode(ListNode node) {
        node.val = node.next.val;
        node.next = node.next.next;
    }
}
```

### [24.Leetcode328奇偶链表](https://leetcode-cn.com/problems/odd-even-linked-list/)

给定一个单链表，把所有的奇数节点和偶数节点分别排在一起。请注意，这里的奇数节点和偶数节点指的是节点编号的奇偶性，而不是节点的值的奇偶性。

请尝试使用原地算法完成。你的算法的空间复杂度应为 O(1)，时间复杂度应为 O(nodes)，nodes 为节点总数。

示例 1:

输入: 1->2->3->4->5->NULL
输出: 1->3->5->2->4->NULL
示例 2:

输入: 2->1->3->5->6->4->7->NULL 
输出: 2->3->6->7->1->5->4->NULL

> 注意截止条件是：不是到null

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode oddEvenList(ListNode head) {
        if(head==null){
            return head;
        }
        ListNode first = head;
        ListNode firstHead = head;
        ListNode second = head.next;
        ListNode secondHead = head.next;
        while(first.next!=null&&second.next!=null){
            first.next = second.next;
            first = first.next;
            second.next = first.next;
            second = second.next;
        }
        // 重新连接上
        first.next = secondHead;
        return firstHead;
    }
}
```

### [25.Leetcode430 扁平化多级双向链表](https://leetcode-cn.com/problems/flatten-a-multilevel-doubly-linked-list/)

多级双向链表中，除了指向下一个节点和前一个节点指针之外，它还有一个子链表指针，可能指向单独的双向链表。这些子列表也可能会有一个或多个自己的子项，依此类推，生成多级数据结构，如下面的示例所示。

给你位于列表第一级的头节点，请你扁平化列表，使所有结点出现在单级双链表中。

 

示例 1：

输入：head = [1,2,3,4,5,6,null,null,null,7,8,9,10,null,null,11,12]
输出：[1,2,3,7,8,11,12,9,10,4,5,6]
解释：

输入的多级列表如下图所示：

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/10/12/multilevellinkedlist.png)

扁平化后的链表如下图：

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/10/12/multilevellinkedlistflattened.png)

> 解题思路：类似于层序遍历

```java
/*
// Definition for a Node.
class Node {
    public int val;
    public Node prev;
    public Node next;
    public Node child;
};
*/

class Solution {
    public Node flatten(Node head) {
        if(head==null){
            return null;
        }
        // 加个哑结点
        Node dummy = new Node(-1,null,head,null);
        // 为了连接
        Node prev = dummy;
        // 用栈
        Stack<Node> stack = new Stack<>();
        stack.push(head);
        while(!stack.isEmpty()){
            Node cur = stack.pop();

            // 连接
            prev.next = cur;
            cur.prev  = prev;

            if(cur.next!=null){
                stack.push(cur.next);
            }

            if(cur.child!=null){
                stack.push(cur.child);
                // 置为空
                cur.child = null;
            }

            prev = cur;
        }

        dummy.next.prev = null;
        return dummy.next;
    }
}
```

### [26.Leetcode445 两数相加](https://leetcode-cn.com/problems/add-two-numbers-ii/)

给你两个 非空 链表来代表两个非负整数。数字最高位位于链表开始位置。它们的每个节点只存储一位数字。将这两数相加会返回一个新的链表。

你可以假设除了数字 0 之外，这两个数字都不会以零开头。

 

进阶：

如果输入链表不能修改该如何处理？换句话说，你不能对列表中的节点进行翻转。

 

示例：

输入：(7 -> 2 -> 4 -> 3) + (5 -> 6 -> 4)
输出：7 -> 8 -> 0 -> 7

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        // 用栈
        Stack<Integer> s1 = new Stack<>();
        Stack<Integer> s2 = new Stack<>();
        while(l1!=null){
            s1.push(l1.val);
            l1 = l1.next;
        }
        while(l2!=null){
            s2.push(l2.val);
            l2 = l2.next;
        }
        // 开始相加
        int rem = 0;
        // 结果
        ListNode l = null;
        while(!s1.isEmpty()||!s2.isEmpty()){
            int t1 = s1.isEmpty()?0:s1.pop();
            int t2 = s2.isEmpty()?0:s2.pop();
            int temp = t1+t2+rem;
            rem = temp/10;
            ListNode node = new ListNode(temp%10);
            node.next = l;
            l = node;
        }
        if(rem!=0){
            ListNode node = new ListNode(rem);
            node.next = l;
            l = node;
        }
        return l;

    }
}
```

### [27.Leetcode707 设计链表](https://leetcode-cn.com/problems/design-linked-list/)

设计链表的实现。您可以选择使用单链表或双链表。单链表中的节点应该具有两个属性：val 和 next。val 是当前节点的值，next 是指向下一个节点的指针/引用。如果要使用双向链表，则还需要一个属性 prev 以指示链表中的上一个节点。假设链表中的所有节点都是 0-index 的。

在链表类中实现这些功能：

get(index)：获取链表中第 index 个节点的值。如果索引无效，则返回-1。
addAtHead(val)：在链表的第一个元素之前添加一个值为 val 的节点。插入后，新节点将成为链表的第一个节点。
addAtTail(val)：将值为 val 的节点追加到链表的最后一个元素。
addAtIndex(index,val)：在链表中的第 index 个节点之前添加值为 val  的节点。如果 index 等于链表的长度，则该节点将附加到链表的末尾。如果 index 大于链表长度，则不会插入节点。如果index小于0，则在头部插入节点。
deleteAtIndex(index)：如果索引 index 有效，则删除链表中的第 index 个节点。


示例：

MyLinkedList linkedList = new MyLinkedList();
linkedList.addAtHead(1);
linkedList.addAtTail(3);
linkedList.addAtIndex(1,2);   //链表变为1-> 2-> 3
linkedList.get(1);            //返回2
linkedList.deleteAtIndex(1);  //现在链表是1-> 3
linkedList.get(1);            //返回3

```java
// 一个单链表
class ListNode{
    int val;
    ListNode next;
    public ListNode(int val){
        this.val = val;
    }
    public ListNode(int val,ListNode next){
        this.val = val;
        this.next = next;
    }
}
class MyLinkedList {
    // 哑结点
    ListNode head;
    int size;
    /** Initialize your data structure here. */
    public MyLinkedList() {
        head = new ListNode(-1);
        size = 0;
    }
    
    /** Get the value of the index-th node in the linked list. If the index is invalid, return -1. */
    public int get(int index) {
        if(index<0||index>=size){
            return -1;
        }
        ListNode cur = head.next;
        // 获取其值
        for(int i=0;i<index;i++){
            cur = cur.next;
        }
        return cur.val;
    }
    
    /** Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list. */
    public void addAtHead(int val) {
        addAtIndex(0,val);
    }
    
    /** Append a node of value val to the last element of the linked list. */
    public void addAtTail(int val) {
        addAtIndex(size,val);
    }   
    
    /** Add a node of value val before the index-th node in the linked list. If index equals to the length of linked list, the node will be appended to the end of linked list. If index is greater than the length, the node will not be inserted. */
    public void addAtIndex(int index, int val) {
        // 往对应的索引下加值
        if(index>size){
            return;
        }
        if(index<0){
            index = 0;
        }
        // 添加其值
        ListNode prev = head;
        for(int i=0;i<index;i++){
            prev = prev.next;
        }
        ListNode cur = new ListNode(val);
        // 先连接后面的
        cur.next = prev.next;
        prev.next = cur;
        // 记录
        size++;
    }
    
    /** Delete the index-th node in the linked list, if the index is valid. */
    public void deleteAtIndex(int index) {
        //删除索引
        if(index<0 || index>=size){
            return;
        }
        ListNode prev = head;
        for(int i=0;i<index;i++){
            prev = prev.next;
        }
        prev.next = prev.next.next;
        // 
        size--;
    }
}

/**
 * Your MyLinkedList object will be instantiated and called as such:
 * MyLinkedList obj = new MyLinkedList();
 * int param_1 = obj.get(index);
 * obj.addAtHead(val);
 * obj.addAtTail(val);
 * obj.addAtIndex(index,val);
 * obj.deleteAtIndex(index);
 */
```

### [28.Leetcode876 链表的中间结点](https://leetcode-cn.com/problems/middle-of-the-linked-list/)

给定一个头结点为 head 的非空单链表，返回链表的中间结点。

如果有两个中间结点，则返回第二个中间结点。

 

示例 1：

输入：[1,2,3,4,5]
输出：此列表中的结点 3 (序列化形式：[3,4,5])
返回的结点值为 3 。 (测评系统对该结点序列化表述是 [3,4,5])。
注意，我们返回了一个 ListNode 类型的对象 ans，这样：
ans.val = 3, ans.next.val = 4, ans.next.next.val = 5, 以及 ans.next.next.next = NULL.
示例 2：

输入：[1,2,3,4,5,6]
输出：此列表中的结点 4 (序列化形式：[4,5,6])
由于该列表有两个中间结点，值分别为 3 和 4，我们返回第二个结点。

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode middleNode(ListNode head) {
        ListNode slow = head;
        ListNode fast = head;
        while(fast!=null&&fast.next!=null){
            slow = slow.next;
            fast = fast.next.next;
        }
        return slow;
    }
}
```

### [29.Leetcode725 分隔链表](https://leetcode-cn.com/problems/split-linked-list-in-parts/)

给定一个头结点为 root 的链表, 编写一个函数以将链表分隔为 k 个连续的部分。

每部分的长度应该尽可能的相等: 任意两部分的长度差距不能超过 1，也就是说可能有些部分为 null。

这k个部分应该按照在链表中出现的顺序进行输出，并且排在前面的部分的长度应该大于或等于后面的长度。

返回一个符合上述规则的链表的列表。

举例： 1->2->3->4, k = 5 // 5 结果 [ [1], [2], [3], [4], null ]

示例 1：

输入: 
root = [1, 2, 3], k = 5
输出: [[1],[2],[3],[],[]]
解释:
输入输出各部分都应该是链表，而不是数组。
例如, 输入的结点 root 的 val= 1, root.next.val = 2, \root.next.next.val = 3, 且 root.next.next.next = null。
第一个输出 output[0] 是 output[0].val = 1, output[0].next = null。
最后一个元素 output[4] 为 null, 它代表了最后一个部分为空链表。

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode[] splitListToParts(ListNode root, int k) {
        // 结果数组
        ListNode[] res = new ListNode[k];
        ListNode head = root;
        int len = 0;
        while(head!=null){
            len++;
            head = head.next;
        }
        // 剩余的
        int rem = len%k;
        int size = len/k;
        head = root;
        // 对其结果赋值了
        for(int i=0;i<k;i++){
            ListNode dummy = new ListNode(-1);
            ListNode node = dummy;
            for(int j=0;j<size+(i<rem?1:0);j++){
                node.next = new ListNode(head.val);
                node = node.next;
                head = head.next;
            }
            res[i] = dummy.next;
        }
        return res;
    }
}
```

### [30.Leetcode817 链表组件](https://leetcode-cn.com/problems/linked-list-components/)

给定链表头结点 head，该链表上的每个结点都有一个 唯一的整型值 。

同时给定列表 G，该列表是上述链表中整型值的一个子集。

返回列表 G 中组件的个数，这里对组件的定义为：链表中一段最长连续结点的值（该值必须在列表 G 中）构成的集合。

 

示例 1：

输入: 
head: 0->1->2->3
G = [0, 1, 3]
输出: 2
解释: 
链表中,0 和 1 是相连接的，且 G 中不包含 2，所以 [0, 1] 是 G 的一个组件，同理 [3] 也是一个组件，故返回 2。

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public int numComponents(ListNode head, int[] nums) {
        // 将数组nums中的值存放在HashSet中
        HashSet<Integer> hashset = new HashSet<>();
        for(int num:nums){
            hashset.add(num);
        }
        // 对head进行遍历
        int res = 0;
        while(head!=null){
            if(hashset.contains(head.val)&&( (head.next==null) || (!hashset.contains(head.next.val)) )){
                res++;
            }
            head = head.next;
        }
        return res;
    }
}
```

### [31.Leetcode1019 链表中的下一个更大节点](https://leetcode-cn.com/problems/next-greater-node-in-linked-list/)

给出一个以头节点 head 作为第一个节点的链表。链表中的节点分别编号为：node_1, node_2, node_3, ... 。

每个节点都可能有下一个更大值（next larger value）：对于 node_i，如果其 next_larger(node_i) 是 node_j.val，那么就有 j > i 且  node_j.val > node_i.val，而 j 是可能的选项中最小的那个。如果不存在这样的 j，那么下一个更大值为 0 。

返回整数答案数组 answer，其中 answer[i] = next_larger(node_{i+1}) 。

注意：在下面的示例中，诸如 [2,1,5] 这样的输入（不是输出）是链表的序列化表示，其头节点的值为 2，第二个节点值为 1，第三个节点值为 5 。

 

示例 1：

输入：[2,1,5]
输出：[5,5,0]
示例 2：

输入：[2,7,4,3,5]
输出：[7,0,5,5,0]

> 解题思路：单调栈

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public int[] nextLargerNodes(ListNode head) {
        // 将其转为数组中的值
        Stack<Integer> stack_num = new Stack<>();
        ListNode cur = head;
        while(cur!=null){
            stack_num.push(cur.val);
            cur = cur.next;
        }
        // 为了保存更大的值逆序来看 需要一个单调栈来存储
        Stack<Integer> stack_max = new Stack<>();
        // 结果
        int[] res = new int[stack_num.size()];
        int index = stack_num.size()-1;
        while(!stack_num.isEmpty()){
            // 往外排
            while(!stack_max.isEmpty()&&stack_num.peek()>=stack_max.peek()){
                stack_max.pop();
            }
            // 存入
            res[index--] = stack_max.isEmpty()?0:stack_max.peek();
            stack_max.push(stack_num.pop());
        }
        return res;
    }
}
```

### [32.Leetcode1171 从链表中删去总和值为零的连续节点](https://leetcode-cn.com/problems/remove-zero-sum-consecutive-nodes-from-linked-list/)

给你一个链表的头节点 head，请你编写代码，反复删去链表中由 总和 值为 0 的连续节点组成的序列，直到不存在这样的序列为止。

删除完毕后，请你返回最终结果链表的头节点。

 

你可以返回任何满足题目要求的答案。

（注意，下面示例中的所有序列，都是对 ListNode 对象序列化的表示。）

示例 1：

输入：head = [1,2,-3,3,1]
输出：[3,1]
提示：答案 [1,2,1] 也是正确的。
示例 2：

输入：head = [1,2,3,-3,4]
输出：[1,2,4]

> 用HashMap结合前缀和的思路来解题



```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode removeZeroSumSublists(ListNode head) {
        // HashMap结合前缀和
        // 增加一个哑结点
        ListNode dummy = new ListNode(0,head);
        ListNode l = dummy;
        // 前缀和+hashMap
        int preSum = 0;
        HashMap<Integer,ListNode> hashMap = new HashMap<>();
        // 第一次遍历
        while(l!=null){
            preSum += l.val;
            hashMap.put(preSum,l);
            // 继续走
            l = l.next;
        }
        // 第二次遍历
        preSum = 0;
        l = dummy;
        while(l!=null){
            preSum += l.val;
            if(hashMap.containsKey(preSum)){
                l.next = hashMap.get(preSum).next;
            }
            l = l.next;
        }
        return dummy.next;
    }
}
```

### [33.Leetcode1290 二进制链表转整数](https://leetcode-cn.com/problems/convert-binary-number-in-a-linked-list-to-integer/)

给你一个单链表的引用结点 head。链表中每个结点的值不是 0 就是 1。已知此链表是一个整数数字的二进制表示形式。

请你返回该链表所表示数字的 十进制值 。

 

示例 1：

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2019/12/15/graph-1.png)

输入：head = [1,0,1]
输出：5
解释：二进制数 (101) 转化为十进制数 (5)

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public int getDecimalValue(ListNode head) {
        int res = 0;
        // 直接存
        Stack<Integer> stack = new Stack<>();
        while(head!=null){
            stack.push(head.val);
            head = head.next;
        }
        // 记录第几次了
        int index = 0;
        while(!stack.isEmpty()){
            int temp = stack.pop()*(int)Math.pow(2,index);
            res += temp;
            index++;
        }
        return res;
    }
}
```

### [34.Leetcode1367 二叉树中的列表](https://leetcode-cn.com/problems/linked-list-in-binary-tree/)

给你一棵以 root 为根的二叉树和一个 head 为第一个节点的链表。

如果在二叉树中，存在一条一直向下的路径，且每个点的数值恰好一一对应以 head 为首的链表中每个节点的值，那么请你返回 True ，否则返回 False 。

一直向下的路径的意思是：从树中某个节点开始，一直连续向下的路径。

 

示例 1：

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/02/29/sample_1_1720.png)

输入：head = [4,2,8], root = [1,4,4,null,2,2,null,1,null,6,8,null,null,null,null,1,3]
输出：true
解释：树中蓝色的节点构成了与链表对应的子路径。

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
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
    public boolean isSubPath(ListNode head, TreeNode root) {
        // 两次调用
        // 如果head为空则为true
        if(head==null){
            return true;
        }
        // head遍历结束了但是root并没有为空
        if(root==null){
            return false;
        }
        // 检查当前的相同则检查
        if((head.val==root.val)&&check(head,root)){
            return true;
        }
        // 如果当前不可以就走其余的
        return isSubPath(head,root.left)||isSubPath(head,root.right);
    }


    // 检查
    public boolean check(ListNode head,TreeNode root){
        if(head==null){
            return true;
        }
        if(root==null){
            return false;
        }
        // 如果不相等
        if(head.val!=root.val){
            return false;
        }
        // 继续判断
        return check(head.next,root.left) || check(head.next,root.right);
    }

}
```

### [35.Leetcode1669 合并两个链表](https://leetcode-cn.com/problems/merge-in-between-linked-lists/)

给你两个链表 list1 和 list2 ，它们包含的元素分别为 n 个和 m 个。

请你将 list1 中第 a 个节点到第 b 个节点删除，并将list2 接在被删除节点的位置。

下图中蓝色边和节点展示了操作后的结果：


请你返回结果链表的头指针。

 ![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/11/28/fig1.png)

示例 1：

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/11/28/merge_linked_list_ex1.png)

输入：list1 = [0,1,2,3,4,5], a = 3, b = 4, list2 = [1000000,1000001,1000002]
输出：[0,1,2,1000000,1000001,1000002,5]
解释：我们删除 list1 中第三和第四个节点，并将 list2 接在该位置。上图中蓝色的边和节点为答案链表。

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode mergeInBetween(ListNode list1, int a, int b, ListNode list2) {
        // 哑结点
        ListNode dummy = new ListNode(-1,list1);
        ListNode l = dummy;
        // 开始走
        ListNode pre = l;
        for(int i=0;i<a;i++){
            pre = pre.next;
        }   
        // 再接着走
        ListNode tail = pre;
        for(int i=a;i<=b;i++){
            tail = tail.next;
        }
        ListNode next = tail.next;
        // 重新连接
        pre.next = list2;
        while(list2.next!=null){
            list2 = list2.next;
        }
        list2.next = next;
        return dummy.next;
    }
}
```

### [36.Leetcode1670 设计前中后队列](https://leetcode-cn.com/problems/design-front-middle-back-queue/)

请你设计一个队列，支持在前，中，后三个位置的 push 和 pop 操作。

请你完成 FrontMiddleBack 类：

FrontMiddleBack() 初始化队列。
void pushFront(int val) 将 val 添加到队列的 最前面 。
void pushMiddle(int val) 将 val 添加到队列的 正中间 。
void pushBack(int val) 将 val 添加到队里的 最后面 。
int popFront() 将 最前面 的元素从队列中删除并返回值，如果删除之前队列为空，那么返回 -1 。
int popMiddle() 将 正中间 的元素从队列中删除并返回值，如果删除之前队列为空，那么返回 -1 。
int popBack() 将 最后面 的元素从队列中删除并返回值，如果删除之前队列为空，那么返回 -1 。
请注意当有 两个 中间位置的时候，选择靠前面的位置进行操作。比方说：

将 6 添加到 [1, 2, 3, 4, 5] 的中间位置，结果数组为 [1, 2, 6, 3, 4, 5] 。
从 [1, 2, 3, 4, 5, 6] 的中间位置弹出元素，返回 3 ，数组变为 [1, 2, 4, 5, 6] 。


示例 1：

输入：
["FrontMiddleBackQueue", "pushFront", "pushBack", "pushMiddle", "pushMiddle", "popFront", "popMiddle", "popMiddle", "popBack", "popFront"]
[[], [1], [2], [3], [4], [], [], [], [], []]
输出：
[null, null, null, null, null, 1, 3, 4, 2, -1]

解释：
FrontMiddleBackQueue q = new FrontMiddleBackQueue();
q.pushFront(1);   // [1]
q.pushBack(2);    // [1, 2]
q.pushMiddle(3);  // [1, 3, 2]
q.pushMiddle(4);  // [1, 4, 3, 2]
q.popFront();     // 返回 1 -> [4, 3, 2]
q.popMiddle();    // 返回 3 -> [4, 2]
q.popMiddle();    // 返回 4 -> [2]
q.popBack();      // 返回 2 -> []
q.popFront();     // 返回 -1 -> [] （队列为空）

```java
class FrontMiddleBackQueue {
    // 用一个arrayList来实现
    List<Integer> list;
    public FrontMiddleBackQueue() {
         list = new ArrayList<>();
    }   
    // 将val添加到队列的最前面
    public void pushFront(int val) {
        list.add(0,val);
    }
    // 将val添加到队列的正中间
    public void pushMiddle(int val) {
        int mid = list.size()/2;
        list.add(mid,val);
    }
    // 将val添加队列的最后面
    public void pushBack(int val) {
        int index = list.size();
        list.add(index,val);
    }
    // 将最前面的元素从队列中删除并返回值
    public int popFront() {
        if(list.isEmpty()){
            return -1;
        }
        int index = 0;
        int temp = list.get(index);
        list.remove(index);
        return temp;
    }
    
    public int popMiddle() {
        if(list.isEmpty()){
            return -1;
        }
        int index = 0;
        if(list.size()%2==0){
            index = list.size()/2-1;
        }else{
            index = list.size()/2;
        }
        int temp = list.get(index);
        list.remove(index);
        return temp;
    }
    
    public int popBack() {
        if(list.isEmpty()){
            return -1;
        }
        int index = list.size()-1;
        int temp = list.get(index);
        list.remove(index);
        return temp;
    }
}

/**
 * Your FrontMiddleBackQueue object will be instantiated and called as such:
 * FrontMiddleBackQueue obj = new FrontMiddleBackQueue();
 * obj.pushFront(val);
 * obj.pushMiddle(val);
 * obj.pushBack(val);
 * int param_4 = obj.popFront();
 * int param_5 = obj.popMiddle();
 * int param_6 = obj.popBack();
 */
```

### [37.Leetcode1721 交换链表中的节点](https://leetcode-cn.com/problems/swapping-nodes-in-a-linked-list/)

给你链表的头节点 head 和一个整数 k 。

交换 链表正数第 k 个节点和倒数第 k 个节点的值后，返回链表的头节点（链表 从 1 开始索引）。

 

示例 1：

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2021/01/10/linked1.jpg)


输入：head = [1,2,3,4,5], k = 2
输出：[1,4,3,2,5]

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode swapNodes(ListNode head, int k) {
        // 记录
        ListNode slow = head;
        ListNode fast = head;
        // 准确定位
        ListNode first = null;
        ListNode second = null;
        for(int i=1;i<k;i++){
            fast = fast.next;
        }
        first = fast;
        while(fast.next!=null){
            fast = fast.next;
            slow = slow.next;
        }
        second = slow;
        // 交换
        int temp = first.val;
        first.val = second.val;
        second.val = temp;
        return head;
    }
}
```

### [38.剑指Ofer06 从尾到头打印链表](https://leetcode-cn.com/problems/cong-wei-dao-tou-da-yin-lian-biao-lcof/)

输入一个链表的头节点，从尾到头反过来返回每个节点的值（用数组返回）。

 

**示例 1：**

```
输入：head = [1,3,2]
输出：[2,3,1]
```

> 用栈来逆序



```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public int[] reversePrint(ListNode head) {
        // 用栈来存储即可
        Stack<Integer> stack = new Stack<>();
        while(head!=null){
            stack.push(head.val);
            head = head.next;
        }
        int[] res = new int[stack.size()];
        int index = 0;
        while(!stack.isEmpty()){
            res[index++] = stack.pop();
        }
        return res;
    }
}
```

### [39.剑指Offfer18 删除链表的节点](https://leetcode-cn.com/problems/shan-chu-lian-biao-de-jie-dian-lcof/)

给定单向链表的头指针和一个要删除的节点的值，定义一个函数删除该节点。

返回删除后的链表的头节点。

注意：此题对比原题有改动

示例 1:

输入: head = [4,5,1,9], val = 5
输出: [4,1,9]
解释: 给定你链表中值为 5 的第二个节点，那么在调用了你的函数之后，该链表应变为 4 -> 1 -> 9.

示例 2:

输入: head = [4,5,1,9], val = 1
输出: [4,5,9]
解释: 给定你链表中值为 1 的第三个节点，那么在调用了你的函数之后，该链表应变为 4 -> 5 -> 9.

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode deleteNode(ListNode head, int val) {
        ListNode dummy = new ListNode(-1,head);
        ListNode cur = dummy;
        while(cur.next!=null){
            if(cur.next.val==val){
                cur.next = cur.next.next;
            }else{
                cur = cur.next;
            }
        }
        return dummy.next;
    }
}
```

### [40.剑指Offer22 链表中倒数第k个节点](https://leetcode-cn.com/problems/lian-biao-zhong-dao-shu-di-kge-jie-dian-lcof/)

输入一个链表，输出该链表中倒数第k个节点。为了符合大多数人的习惯，本题从1开始计数，即链表的尾节点是倒数第1个节点。

例如，一个链表有 6 个节点，从头节点开始，它们的值依次是 1、2、3、4、5、6。这个链表的倒数第 3 个节点是值为 4 的节点。

 

示例：

给定一个链表: 1->2->3->4->5, 和 k = 2.

返回链表 4->5.

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode getKthFromEnd(ListNode head, int k) {
        ListNode slow = head;
        ListNode fast = head;
        for(int i=1;i<k;i++){
            fast = fast.next;
        }
        while(fast.next!=null){
            slow = slow.next;
            fast = fast.next;
        }
        return slow;
    }
}
```

### [41.剑指Offer24 反转链表](https://leetcode-cn.com/problems/fan-zhuan-lian-biao-lcof/)


定义一个函数，输入一个链表的头节点，反转该链表并输出反转后链表的头节点。

 

**示例:**

```
输入: 1->2->3->4->5->NULL
输出: 5->4->3->2->1->NULL
```

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode reverseList(ListNode head) {
        ListNode pre = null;
        ListNode next = null;
        while(head!=null){
            next = head.next;
            head.next = pre;
            pre = head;
            head = next;
        }
        return pre;
    }
}
```

### [42.剑指Offer35 复杂链表的复制](https://leetcode-cn.com/problems/fu-za-lian-biao-de-fu-zhi-lcof/)

请实现 copyRandomList 函数，复制一个复杂链表。在复杂链表中，每个节点除了有一个 next 指针指向下一个节点，还有一个 random 指针指向链表中的任意节点或者 null。

 

示例 1：

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/01/09/e1.png)

输入：head = [[7,null],[13,0],[11,4],[10,2],[1,0]]
输出：[[7,null],[13,0],[11,4],[10,2],[1,0]]

```java
/*
// Definition for a Node.
class Node {
    int val;
    Node next;
    Node random;

    public Node(int val) {
        this.val = val;
        this.next = null;
        this.random = null;
    }
}
*/
class Solution {
    public Node copyRandomList(Node head) {
        // 复制链表用HashMap
        HashMap<Node,Node> hashMap = new HashMap<>();
        // 先挨个复制
        Node node = head;
        while(node!=null){
            hashMap.put(node,new Node(node.val));
            node = node.next;
        }
        // 接着
        node = head;
        while(node!=null){
            hashMap.get(node).next = hashMap.get(node.next);
            hashMap.get(node).random = hashMap.get(node.random);
            node = node.next;
        }
        return hashMap.get(head);
    }
}
```

### [43.剑指Offer52 两个链表的第一个公共节点](https://leetcode-cn.com/problems/liang-ge-lian-biao-de-di-yi-ge-gong-gong-jie-dian-lcof/)

输入两个链表，找出它们的第一个公共节点。

如下面的两个链表：

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/14/160_statement.png)

在节点 c1 开始相交。

 

示例 1：

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/14/160_example_1.png)

输入：intersectVal = 8, listA = [4,1,8,4,5], listB = [5,0,1,8,4,5], skipA = 2, skipB = 3
输出：Reference of the node with value = 8
输入解释：相交节点的值为 8 （注意，如果两个列表相交则不能为 0）。从各自的表头开始算起，链表 A 为 [4,1,8,4,5]，链表 B 为 [5,0,1,8,4,5]。在 A 中，相交节点前有 2 个节点；在 B 中，相交节点前有 3 个节点。

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) {
 *         val = x;
 *         next = null;
 *     }
 * }
 */
public class Solution {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode newHeadA = headA;
        ListNode newHeadB = headB;
        while(headA!=headB){
            headA = headA==null?newHeadB:headA.next;
            headB = headB==null?newHeadA:headB.next;
        }
        return headA;
    }
}
```

### [44.面试题02.01 移除重复节点](https://leetcode-cn.com/problems/remove-duplicate-node-lcci/)

编写代码，移除未排序链表中的重复节点。保留最开始出现的节点。

示例1:

 输入：[1, 2, 3, 3, 2, 1]
 输出：[1, 2, 3]
示例2:

 输入：[1, 1, 1, 1, 2]
 输出：[1, 2]
提示：

链表长度在[0, 20000]范围内。
链表元素在[0, 20000]范围内。
进阶：

> 解题思路：用hashset来解决

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    // 移除未排序链表中的重复节点，保留最开始出现的结点
    public ListNode removeDuplicateNodes(ListNode head) {
        // 用hashset来去重
        HashSet<Integer> hashset = new HashSet<>();
        // 新链表
        ListNode dummy = new ListNode(-1);
        dummy.next = head;
        // 指针
        ListNode l = dummy;
        // 对其遍历
        while(head!=null){
            //判断
            if(!hashset.contains(head.val)){
                // hashset为空或者不包含该值
                hashset.add(head.val);
                l.next = head;
                l = l.next;
                head = head.next;
            }else{
                // 包含该值
               head = head.next;
               l.next = head;
            }
        }
        return dummy.next;
    }
}
```

### [45.面试题02.02 返回倒数第k个结点](https://leetcode-cn.com/problems/kth-node-from-end-of-list-lcci/)

实现一种算法，找出单向链表中倒数第 k 个节点。返回该节点的值。

注意：本题相对原题稍作改动

示例：

输入： 1->2->3->4->5 和 k = 2
输出： 4

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public int kthToLast(ListNode head, int k) {
        ListNode slow = head;
        ListNode fast = head;
        for(int i=1;i<k;i++){
            fast = fast.next;
        }
        while(fast.next!=null){
            fast = fast.next;
            slow = slow.next;
        }
        return slow.val;
    }
}
```

### [46.面试题02.03 删除中间节点](https://leetcode-cn.com/problems/delete-middle-node-lcci/)

若链表中的某个节点，既不是链表头节点，也不是链表尾节点，则称其为该链表的「中间节点」。

假定已知链表的某一个中间节点，请实现一种算法，将该节点从链表中删除。

例如，传入节点 c（位于单向链表 a->b->c->d->e->f 中），将其删除后，剩余链表为 a->b->d->e->f

 

示例：

输入：节点 5 （位于单向链表 4->5->1->9 中）
输出：不返回任何数据，从链表中删除传入的节点 5，使链表变为 4->1->9。



```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public void deleteNode(ListNode node) {
        //删除当前的结点
        node.val = node.next.val;
        node.next = node.next.next;
    }
}
```

### [47.面试题02.04 分割链表](https://leetcode-cn.com/problems/partition-list-lcci/)

编写程序以 x 为基准分割链表，使得所有小于 x 的节点排在大于或等于 x 的节点之前。如果链表中包含 x，x 只需出现在小于 x 的元素之后(如下所示)。分割元素 x 只需处于“右半部分”即可，其不需要被置于左右两部分之间。

示例:

输入: head = 3->5->8->5->10->2->1, x = 5
输出: 3->1->2->10->5->5->8

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    // 分割链表
    public ListNode partition(ListNode head, int x) {
        ListNode small = new ListNode(-1);
        ListNode big   = new ListNode(-1);
        ListNode smallHead = small;
        ListNode bigHead = big;
        // 对其遍历
        while(head!=null){
            //判断
            if(head.val<x){
                small.next = head;
                small = small.next;
            }else{
                big.next = head;
                big = big.next;
            }
            head = head.next;
        }
        // 合并
        big.next = null;
        // 环绕圈的原因是忘了给最后加null
        small.next = bigHead.next;
        return smallHead.next;
    }
}
```

### [48.面试题02.05 链表求和](https://leetcode-cn.com/problems/sum-lists-lcci/)

给定两个用链表表示的整数，每个节点包含一个数位。

这些数位是反向存放的，也就是个位排在链表首部。

编写函数对这两个整数求和，并用链表形式返回结果。

 

示例：

输入：(7 -> 1 -> 6) + (5 -> 9 -> 2)，即617 + 295
输出：2 -> 1 -> 9，即912
进阶：思考一下，假设这些数位是正向存放的，又该如何解决呢?

示例：

输入：(6 -> 1 -> 7) + (2 -> 9 -> 5)，即617 + 295
输出：9 -> 1 -> 2，即912

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        // 求和
        int remain = 0;
        // 新链表
        ListNode dummy = new ListNode(-1);
        ListNode l = dummy;
        while(l1!=null||l2!=null){
            int n1 = l1==null?0:l1.val;
            int n2 = l2==null?0:l2.val;
            int sum = n1+n2+remain;
            remain = sum/10;
            int val = sum%10;
            l.next = new ListNode(val);
            l = l.next;

            if(l1!=null){
                l1 = l1.next;
            }
            if(l2!=null){
                l2 = l2.next;
            }
        }
        if(remain!=0){
            l.next = new ListNode(remain);
        }

        return dummy.next;
    }   
}
```

### [49.面试题02.06 回文链表](https://leetcode-cn.com/problems/palindrome-linked-list-lcci/)

编写一个函数，检查输入的链表是否是回文的。

 

示例 1：

输入： 1->2
输出： false 
示例 2：

输入： 1->2->2->1
输出： true 

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    // 检查链表是否是回文
    public boolean isPalindrome(ListNode head) {
        if(head==null || head.next==null){
            return true;
        }
        ListNode medium = findMedium(head);
        // 后面翻转
        ListNode next = medium.next;
        medium.next = null;
        next = reverse(next);
        // 开始判断
        return judge(next,head);
    }

    // 找中间的链表中间结点
    public ListNode findMedium(ListNode head){
        ListNode slow = head;
        ListNode fast = head;
        while(fast.next!=null&&fast.next.next!=null){
            slow = slow.next;
            fast = fast.next.next;
        }
        return slow;
    }
    // 翻转链表
    public ListNode reverse(ListNode head){
        ListNode pre = null;
        ListNode next = null;
        while(head!=null){
            next = head.next;
            head.next = pre;
            pre = head;
            head = next;
        }
        return pre;
    }
    // 判断
    public boolean judge(ListNode headA,ListNode headB){
        while(headA!=null&&headB!=null){
            if(headA.val!=headB.val){
                return false;
            }
            headA = headA.next;
            headB = headB.next;
        }
        return true;
    }
    
}
```

### [50.面试题02.07 链表相交](https://leetcode-cn.com/problems/intersection-of-two-linked-lists-lcci/)



给定两个（单向）链表，判定它们是否相交并返回交点。请注意相交的定义基于节点的引用，而不是基于节点的值。换句话说，如果一个链表的第k个节点与另一个链表的第j个节点是同一节点（引用完全相同），则这两个链表相交。


示例 1：

输入：intersectVal = 8, listA = [4,1,8,4,5], listB = [5,0,1,8,4,5], skipA = 2, skipB = 3
输出：Reference of the node with value = 8
输入解释：相交节点的值为 8 （注意，如果两个列表相交则不能为 0）。从各自的表头开始算起，链表 A 为 [4,1,8,4,5]，链表 B 为 [5,0,1,8,4,5]。在 A 中，相交节点前有 2 个节点；在 B 中，相交节点前有 3 个节点。

示例 2：

输入：intersectVal = 2, listA = [0,9,1,2,4], listB = [3,2,4], skipA = 3, skipB = 1
输出：Reference of the node with value = 2
输入解释：相交节点的值为 2 （注意，如果两个列表相交则不能为 0）。从各自的表头开始算起，链表 A 为 [0,9,1,2,4]，链表 B 为 [3,2,4]。在 A 中，相交节点前有 3 个节点；在 B 中，相交节点前有 1 个节点。

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) {
 *         val = x;
 *         next = null;
 *     }
 * }
 */
public class Solution {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        // 判断两个链表是否相交单向
        ListNode newHeadA = headA;
        ListNode newHeadB = headB;
        while(headA!=headB){
            headA = headA==null?newHeadB:headA.next;
            headB = headB==null?newHeadA:headB.next;
        }
        return headA;
    }
}
```

### [51.面试题02.08 环路检测](https://leetcode-cn.com/problems/linked-list-cycle-lcci/)

给定一个链表，如果它是有环链表，实现一个算法返回环路的开头节点。

如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，我们使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 pos 是 -1，则在该链表中没有环。注意：pos 不作为参数进行传递，仅仅是为了标识链表的实际情况。

 

示例 1：

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/07/circularlinkedlist.png)

输入：head = [3,2,0,-4], pos = 1
输出：tail connects to node index 1
解释：链表中有一个环，其尾部连接到第二个节点。

> 环路快慢指针检测

```java
/**
 * Definition for singly-linked list.
 * class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) {
 *         val = x;
 *         next = null;
 *     }
 * }
 */
public class Solution {
    public ListNode detectCycle(ListNode head) {
        //判断是否有环
        ListNode slow = head;
        ListNode fast = head;
        while(fast!=null&&fast.next!=null){
            fast = fast.next.next;
            slow = slow.next;
            if(slow==fast){
                fast = head;
                // 再次判断
                while(fast!=slow){
                    slow = slow.next;
                    fast = fast.next;
                }
                return fast;
            }
        }   
        return null;
    }
}
```

### [51.Leetcode705 设计哈希集合](https://leetcode-cn.com/problems/design-hashset/)

不使用任何内建的哈希表库设计一个哈希集合（HashSet）。

实现 MyHashSet 类：

void add(key) 向哈希集合中插入值 key 。
bool contains(key) 返回哈希集合中是否存在这个值 key 。
void remove(key) 将给定值 key 从哈希集合中删除。如果哈希集合中没有这个值，什么也不做。

示例：

输入：
["MyHashSet", "add", "add", "contains", "contains", "add", "contains", "remove", "contains"]
[[], [1], [2], [1], [3], [2], [2], [2], [2]]
输出：
[null, null, null, true, false, null, true, null, false]

解释：
MyHashSet myHashSet = new MyHashSet();
myHashSet.add(1);      // set = [1]
myHashSet.add(2);      // set = [1, 2]
myHashSet.contains(1); // 返回 True
myHashSet.contains(3); // 返回 False ，（未找到）
myHashSet.add(2);      // set = [1, 2]
myHashSet.contains(2); // 返回 True
myHashSet.remove(2);   // set = [1]
myHashSet.contains(2); // 返回 False ，（已移除）

> 解题思路：数组

```java
class MyHashSet {
    private static final int capacity = 1000006;
    boolean[] data = new boolean[capacity];
    /** Initialize your data structure here. */
    public MyHashSet() {

    }
    
    public void add(int key) {
        data[key] = true;
    }
    
    public void remove(int key) {
        data[key] = false;
    }
    
    /** Returns true if this set contains the specified element */
    public boolean contains(int key) {
        return data[key];
    }
}

/**
 * Your MyHashSet object will be instantiated and called as such:
 * MyHashSet obj = new MyHashSet();
 * obj.add(key);
 * obj.remove(key);
 * boolean param_3 = obj.contains(key);
 */
```



> 解题思路：链表数组

```java
class MyHashSet {
    private static final int capacity = 1000;
    // 链表数组
    LinkedList[] data;

    /** Initialize your data structure here. */
    public MyHashSet() {
        data = new LinkedList[capacity];
        //初始化操作
        for(int i=0;i<capacity;i++){
            data[i] = new LinkedList<Integer>();
        }
    }
    // 添加
    public void add(int key) {
        //迭代器
        int index = hash(key);
        Iterator<Integer> it = data[index].iterator();
        while(it.hasNext()){
            Integer cur_value = it.next();
            // 判断是否相等
            if(cur_value==key){
                return;
            }
        }
        // 添加进来
        data[index].offerLast(key);
    }
    //删除
    public void remove(int key) {
        //迭代器
        int index = hash(key);
        Iterator<Integer> it = data[index].iterator();
        while(it.hasNext()){
            Integer cur_value = it.next();
            // 判断是否相等
            if(cur_value==key){
                //存在
                data[index].remove(cur_value);
                return;
            }
        }
    }
    
    // 是否包含
    /** Returns true if this set contains the specified element */
    public boolean contains(int key) {
        //迭代器
        int index = hash(key);
        Iterator<Integer> it = data[index].iterator();
        while(it.hasNext()){
            Integer cur_value = it.next();
            // 判断是否相等
            if(cur_value==key){
                return true;
            }
        }
        return false;
    }

    // hash函数
    public int hash(int key){
        return key%capacity;
    }
}

/**
 * Your MyHashSet object will be instantiated and called as such:
 * MyHashSet obj = new MyHashSet();
 * obj.add(key);
 * obj.remove(key);
 * boolean param_3 = obj.contains(key);
 */
```

### [52.Leetcode706 设计哈希映射](https://leetcode-cn.com/problems/design-hashmap/)

不使用任何内建的哈希表库设计一个哈希映射（HashMap）。

实现 MyHashMap 类：

MyHashMap() 用空映射初始化对象
void put(int key, int value) 向 HashMap 插入一个键值对 (key, value) 。如果 key 已经存在于映射中，则更新其对应的值 value 。
int get(int key) 返回特定的 key 所映射的 value ；如果映射中不包含 key 的映射，返回 -1 。
void remove(key) 如果映射中存在 key 的映射，则移除 key 和它所对应的 value 。


示例：

输入：
["MyHashMap", "put", "put", "get", "get", "put", "get", "remove", "get"]
[[], [1, 1], [2, 2], [1], [3], [2, 1], [2], [2], [2]]
输出：
[null, null, null, 1, -1, null, 1, null, -1]

解释：
MyHashMap myHashMap = new MyHashMap();
myHashMap.put(1, 1); // myHashMap 现在为 [[1,1]]
myHashMap.put(2, 2); // myHashMap 现在为 [[1,1], [2,2]]
myHashMap.get(1);    // 返回 1 ，myHashMap 现在为 [[1,1], [2,2]]
myHashMap.get(3);    // 返回 -1（未找到），myHashMap 现在为 [[1,1], [2,2]]
myHashMap.put(2, 1); // myHashMap 现在为 [[1,1], [2,1]]（更新已有的值）
myHashMap.get(2);    // 返回 1 ，myHashMap 现在为 [[1,1], [2,1]]
myHashMap.remove(2); // 删除键为 2 的数据，myHashMap 现在为 [[1,1]]
myHashMap.get(2);    // 返回 -1（未找到），myHashMap 现在为 [[1,1]]

```java
class Node{
    int key;
    int value;
    public Node(){

    }
    public Node(int key,int value){
        this.key = key;
        this.value = value;
    }
}
class MyHashMap {
    //链表数组
    private static final int capacity = 1000;
    LinkedList[] data;
    /** Initialize your data structure here. */
    public MyHashMap() {
        // 初始化
        data = new LinkedList[capacity];
        for(int i=0;i<capacity;i++){
            data[i] = new LinkedList<Node>();
        }
    }
    
    // 存入值
    /** value will always be non-negative. */
    public void put(int key, int value) {
        int i = hash(key);
        // 迭代
        Iterator<Node> it = data[i].iterator();
        while(it.hasNext()){
            Node node = it.next();
            // 判断key
            if(node.key==key){
                node.value = value;
                return;
            }
        } 
        // 没找到
        data[i].offerLast(new Node(key,value));
    }

    // 取值
    /** Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key */
    public int get(int key) {
        int i = hash(key);
        // 迭代
        Iterator<Node> it = data[i].iterator();
        while(it.hasNext()){
            Node node = it.next();
            if(node.key==key){
                return node.value;
            }
        }
        return -1;
    }
    
    // 删除值
    /** Removes the mapping of the specified value key if this map contains a mapping for the key */
    public void remove(int key) {
        int i = hash(key);
        Iterator<Node> it = data[i].iterator();
        while(it.hasNext()){
            Node node = it.next();
            if(node.key==key){
                data[i].remove(node);       
                return;
            }
        }
    }

    // hash函数
    private int hash(int index){
        return index%capacity;
    }
}

/**
 * Your MyHashMap object will be instantiated and called as such:
 * MyHashMap obj = new MyHashMap();
 * obj.put(key,value);
 * int param_2 = obj.get(key);
 * obj.remove(key);
 */
```

### [53.Leetcode1472 设计游览器历史记录](https://leetcode-cn.com/problems/design-browser-history/)

你有一个只支持单个标签页的 浏览器 ，最开始你浏览的网页是 homepage ，你可以访问其他的网站 url ，也可以在浏览历史中后退 steps 步或前进 steps 步。

请你实现 BrowserHistory 类：

BrowserHistory(string homepage) ，用 homepage 初始化浏览器类。
void visit(string url) 从当前页跳转访问 url 对应的页面  。执行此操作会把浏览历史前进的记录全部删除。
string back(int steps) 在浏览历史中后退 steps 步。如果你只能在浏览历史中后退至多 x 步且 steps > x ，那么你只后退 x 步。请返回后退 至多 steps 步以后的 url 。
string forward(int steps) 在浏览历史中前进 steps 步。如果你只能在浏览历史中前进至多 x 步且 steps > x ，那么你只前进 x 步。请返回前进 至多 steps步以后的 url 。


示例：

输入：
["BrowserHistory","visit","visit","visit","back","back","forward","visit","forward","back","back"]
[["leetcode.com"],["google.com"],["facebook.com"],["youtube.com"],[1],[1],[1],["linkedin.com"],[2],[2],[7]]
输出：
[null,null,null,null,"facebook.com","google.com","facebook.com",null,"linkedin.com","google.com","leetcode.com"]

解释：
BrowserHistory browserHistory = new BrowserHistory("leetcode.com");
browserHistory.visit("google.com");       // 你原本在浏览 "leetcode.com" 。访问 "google.com"
browserHistory.visit("facebook.com");     // 你原本在浏览 "google.com" 。访问 "facebook.com"
browserHistory.visit("youtube.com");      // 你原本在浏览 "facebook.com" 。访问 "youtube.com"
browserHistory.back(1);                   // 你原本在浏览 "youtube.com" ，后退到 "facebook.com" 并返回 "facebook.com"
browserHistory.back(1);                   // 你原本在浏览 "facebook.com" ，后退到 "google.com" 并返回 "google.com"
browserHistory.forward(1);                // 你原本在浏览 "google.com" ，前进到 "facebook.com" 并返回 "facebook.com"
browserHistory.visit("linkedin.com");     // 你原本在浏览 "facebook.com" 。 访问 "linkedin.com"
browserHistory.forward(2);                // 你原本在浏览 "linkedin.com" ，你无法前进任何步数。
browserHistory.back(2);                   // 你原本在浏览 "linkedin.com" ，后退两步依次先到 "facebook.com" ，然后到 "google.com" ，并返回 "google.com"
browserHistory.back(7);                   // 你原本在浏览 "google.com"， 你只能后退一步到 "leetcode.com" ，并返回 "leetcode.com"

> 解题思路：用数组，两个指针front和end即可

```java
class BrowserHistory {
    // 用数组就可以记录了
    String[] history;
    int front = -1;
    int end = -1;
    public BrowserHistory(String homepage) {
        history = new String[5001];
        visit(homepage);
    }
    
    public void visit(String url) {
        history[++front] = url;
        end = front;
    }
    // 返回
    public String back(int steps) {
        front -= Math.min(front,steps);
        return history[front];
    }
    // 前进
    public String forward(int steps) {
        front += Math.min(end-front,steps);
        return history[front];
    }
}

/**
 * Your BrowserHistory object will be instantiated and called as such:
 * BrowserHistory obj = new BrowserHistory(homepage);
 * obj.visit(url);
 * String param_2 = obj.back(steps);
 * String param_3 = obj.forward(steps);
 */
```

### [蓄水池抽样算法]

**一、问题**

> 给定一个数据流，数据流长度N很大，且N直到处理完所有数据之前都不可知，请问如何在只遍历一遍数据（O(N)）的情况下，能够随机选取出m个不重复的数据。


**这个场景强调了3件事：**

- 数据流长度N很大且不可知，所以不能一次性存入内存。
- 时间复杂度为O(N)。
- 随机选取m个数，每个数被选中的概率为m/N。

第1点限制了不能直接取N内的m个随机数，然后按索引取出数据。第2点限制了不能先遍历一遍，然后分块存储数据，再随机选取。第3点是数据选取绝对随机的保证。讲真，在不知道蓄水池算法前，我想破脑袋也不知道该题做何解。

**二、核心代码**

蓄水池抽样算法的核心如下：

```css
int[] reservoir = new int[m];

// init
for (int i = 0; i < reservoir.length; i++)
{
    reservoir[i] = dataStream[i];
}

for (int i = m; i < dataStream.length; i++)
{
    // 随机获得一个[0, i]内的随机整数
    int d = rand.nextInt(i + 1);
    // 如果随机整数落在[0, m-1]范围内，则替换蓄水池中的元素
    if (d < m)
    {
        reservoir[d] = dataStream[i];
    }
}
```
注：这里使用已知长度的数组dataStream来表示未知长度的数据流，并假设数据流长度大于蓄水池容量m。

算法思路大致如下：

- 如果接收的数据量小于m，则依次放入蓄水池。
- 当接收到第i个数据时，i >= m，在[0, i]范围内取以随机数d，若d的落在[0, m-1]范围内，则用接收到的第i个数据替换蓄水池中的第d个数据。
- 重复步骤2。

算法的精妙之处在于：当处理完所有的数据时，蓄水池中的每个数据都是以m/N的概率获得的。

### [54.Leetcode382 链表随机结点](https://leetcode-cn.com/problems/linked-list-random-node/)

给定一个单链表，随机选择链表的一个节点，并返回相应的节点值。保证每个节点被选的概率一样。

进阶:
如果链表十分大且长度未知，如何解决这个问题？你能否使用常数级空间复杂度实现？

示例:

// 初始化一个单链表 [1,2,3].
ListNode head = new ListNode(1);
head.next = new ListNode(2);
head.next.next = new ListNode(3);
Solution solution = new Solution(head);

// getRandom()方法应随机返回1,2,3中的一个，保证每个元素被返回的概率相等。
solution.getRandom();

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    ListNode head;
    /** @param head The linked list's head.
        Note that the head is guaranteed to be not null, so it contains at least one node. */
    public Solution(ListNode head) {
        this.head = head;
    }
    
    /** Returns a random node's value. */
    public int getRandom() {
        ListNode cur = head;
        Random r = new Random();
        // 开始计算
        int count = 0;
        int res = head.val;
        // 开始计算
        while(cur!=null){
            if(r.nextInt(++count)==0){
                res = cur.val;
            }
            cur = cur.next;
        }
        return res;
    }
}

/**
 * Your Solution object will be instantiated and called as such:
 * Solution obj = new Solution(head);
 * int param_1 = obj.getRandom();
 */
```

